import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Vector3
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import math
 
 
@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    i_max: float = 10.0
    d_filter_coeff: float = 0.85
 
 
@dataclass
class AxisState:
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    filtered_derivative: float = 0.0
 
 
@dataclass
class VectorControllerConfig:
    joint_names: list = field(default_factory=lambda: ["joint_1", "joint_2"])
    control_rate_hz: float = 500.0
    max_torque: list = field(default_factory=lambda: [15.0, 8.0])
    max_velocity: list = field(default_factory=lambda: [3.14, 3.14])
    position_gains: list = field(default_factory=lambda: [
        PIDGains(kp=120.0, ki=2.5, kd=18.0, i_max=8.0, d_filter_coeff=0.80),
        PIDGains(kp=85.0, ki=1.8, kd=12.0, i_max=5.0, d_filter_coeff=0.80),
    ])
    velocity_gains: list = field(default_factory=lambda: [
        PIDGains(kp=22.0, ki=0.8, kd=1.2, i_max=12.0, d_filter_coeff=0.90),
        PIDGains(kp=16.0, ki=0.6, kd=0.9, i_max=8.0, d_filter_coeff=0.90),
    ])
    gravity_comp_enabled: bool = True
    link_masses: list = field(default_factory=lambda: [1.85, 0.95])
    link_lengths: list = field(default_factory=lambda: [0.35, 0.28])
    g: float = 9.81
    feedforward_enabled: bool = True
    inertia: list = field(default_factory=lambda: [0.042, 0.018])
    damping: list = field(default_factory=lambda: [0.12, 0.08])
 
 
class PIDController:
    def __init__(self, gains: PIDGains):
        self.gains = gains
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self._initialized = False
 
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self._initialized = False
 
    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
 
        if not self._initialized:
            self.prev_error = error
            self._initialized = True
 
        self.integral += error * dt
        self.integral = np.clip(
            self.integral, -self.gains.i_max, self.gains.i_max
        )
 
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (
            self.gains.d_filter_coeff * self.filtered_derivative
            + (1.0 - self.gains.d_filter_coeff) * raw_derivative
        )
 
        output = (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * self.filtered_derivative
        )
 
        self.prev_error = error
        return output
 
 
class GravityCompensator:
    def __init__(self, config: VectorControllerConfig):
        self.masses = config.link_masses
        self.lengths = config.link_lengths
        self.g = config.g
 
    def compute(self, q1: float, q2: float) -> Tuple[float, float]:
        m1, m2 = self.masses
        l1, l2 = self.lengths
        lc1, lc2 = l1 * 0.5, l2 * 0.5
 
        tau2 = m2 * self.g * lc2 * math.cos(q1 + q2)
        tau1 = (m1 * self.g * lc1 + m2 * self.g * l1) * math.cos(q1) + tau2
 
        return tau1, tau2
 
 
class FeedforwardCompensator:
    def __init__(self, config: VectorControllerConfig):
        self.inertia = config.inertia
        self.damping = config.damping
 
    def compute(
        self,
        q_ddot: np.ndarray,
        q_dot: np.ndarray,
    ) -> np.ndarray:
        ff = np.array([
            self.inertia[i] * q_ddot[i] + self.damping[i] * q_dot[i]
            for i in range(2)
        ])
        return ff
 
 
class VelocityProfileGenerator:
    def __init__(self, max_vel: list, max_acc: list = None):
        self.max_vel = np.array(max_vel)
        self.max_acc = np.array(max_acc if max_acc else [4.0, 4.0])
 
    def trapezoidal_step(
        self,
        current: np.ndarray,
        target: np.ndarray,
        current_vel: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        error = target - current
        distance = np.abs(error)
        sign = np.sign(error)
 
        stop_dist = (current_vel ** 2) / (2.0 * self.max_acc + 1e-9)
        decel_needed = distance < stop_dist
 
        desired_vel = np.where(
            decel_needed,
            sign * np.sqrt(2.0 * self.max_acc * np.maximum(distance, 0.0)),
            sign * self.max_vel,
        )
        desired_vel = np.clip(desired_vel, -self.max_vel, self.max_vel)
 
        delta_vel = desired_vel - current_vel
        max_delta = self.max_acc * dt
        clamped_delta = np.clip(delta_vel, -max_delta, max_delta)
        new_vel = current_vel + clamped_delta
 
        new_pos = current + new_vel * dt
        return new_pos, new_vel
 
 
class VectorController2DOF(Node):
    def __init__(self):
        super().__init__("vector_controller_2dof")
 
        self.config = VectorControllerConfig()
        self.dt = 1.0 / self.config.control_rate_hz
 
        self.axis_states = [AxisState(), AxisState()]
        self.position_reference = np.zeros(2)
        self.velocity_reference = np.zeros(2)
        self.current_velocity_cmd = np.zeros(2)
 
        self.pos_pids = [
            PIDController(g) for g in self.config.position_gains
        ]
        self.vel_pids = [
            PIDController(g) for g in self.config.velocity_gains
        ]
 
        self.gravity_comp = GravityCompensator(self.config)
        self.ff_comp = FeedforwardCompensator(self.config)
        self.vel_profile = VelocityProfileGenerator(
            self.config.max_velocity
        )
 
        self.prev_vel_cmd = np.zeros(2)
        self.controller_active = False
        self.last_joint_time = None
 
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
 
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/robot/joint_states",
            self._joint_state_callback,
            qos,
        )
 
        self.position_ref_sub = self.create_subscription(
            Float64MultiArray,
            "/vector_controller/position_reference",
            self._position_reference_callback,
            10,
        )
 
        self.velocity_ref_sub = self.create_subscription(
            Float64MultiArray,
            "/vector_controller/velocity_reference",
            self._velocity_reference_callback,
            10,
        )
 
        self.torque_pub = self.create_publisher(
            Float64MultiArray,
            "/robot/joint_torques",
            qos,
        )
 
        self.controller_state_pub = self.create_publisher(
            Float64MultiArray,
            "/vector_controller/state",
            10,
        )
 
        self.error_pub = self.create_publisher(
            Float64MultiArray,
            "/vector_controller/tracking_error",
            10,
        )
 
        self.control_timer = self.create_timer(
            self.dt, self._control_loop
        )
 
        self.get_logger().info(
            f"VectorController2DOF initialised | "
            f"rate={self.config.control_rate_hz} Hz | "
            f"joints={self.config.joint_names}"
        )
 
    def _joint_state_callback(self, msg: JointState):
        name_to_idx = {
            name: i for i, name in enumerate(msg.name)
        }
        for axis_idx, joint_name in enumerate(self.config.joint_names):
            if joint_name not in name_to_idx:
                continue
            j = name_to_idx[joint_name]
            self.axis_states[axis_idx].position = (
                msg.position[j] if len(msg.position) > j else 0.0
            )
            self.axis_states[axis_idx].velocity = (
                msg.velocity[j] if len(msg.velocity) > j else 0.0
            )
            self.axis_states[axis_idx].effort = (
                msg.effort[j] if len(msg.effort) > j else 0.0
            )
 
        if not self.controller_active:
            self.position_reference = np.array(
                [s.position for s in self.axis_states]
            )
            self.controller_active = True
 
        self.last_joint_time = self.get_clock().now()
 
    def _position_reference_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Position reference must have 2 elements.")
            return
        self.position_reference = np.array(msg.data[:2])
        self.velocity_reference = np.zeros(2)
 
    def _velocity_reference_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Velocity reference must have 2 elements.")
            return
        self.velocity_reference = np.array(msg.data[:2])
 
    def _get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([s.position for s in self.axis_states])
        q_dot = np.array([s.velocity for s in self.axis_states])
        return q, q_dot
 
    def _cascade_control(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
    ) -> np.ndarray:
        pos_error = q_ref - q
        vel_cmd = np.array([
            self.pos_pids[i].compute(pos_error[i], self.dt)
            for i in range(2)
        ])
        vel_cmd = np.clip(vel_cmd, -self.config.max_velocity, self.config.max_velocity)
        vel_cmd += q_dot_ref
 
        vel_error = vel_cmd - q_dot
        torque_cmd = np.array([
            self.vel_pids[i].compute(vel_error[i], self.dt)
            for i in range(2)
        ])
        return torque_cmd, pos_error, vel_error
 
    def _apply_gravity_compensation(
        self, torque_cmd: np.ndarray, q: np.ndarray
    ) -> np.ndarray:
        if not self.config.gravity_comp_enabled:
            return torque_cmd
        tau_g1, tau_g2 = self.gravity_comp.compute(q[0], q[1])
        return torque_cmd + np.array([tau_g1, tau_g2])
 
    def _apply_feedforward(
        self, torque_cmd: np.ndarray, q_dot: np.ndarray
    ) -> np.ndarray:
        if not self.config.feedforward_enabled:
            return torque_cmd
        q_ddot_est = (q_dot - self.prev_vel_cmd) / (self.dt + 1e-9)
        ff = self.ff_comp.compute(q_ddot_est, q_dot)
        return torque_cmd + ff
 
    def _saturate_torque(self, torque_cmd: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(torque_cmd[i], -self.config.max_torque[i], self.config.max_torque[i])
            for i in range(2)
        ])
 
    def _publish_torque(self, torque: np.ndarray):
        msg = Float64MultiArray()
        msg.data = torque.tolist()
        self.torque_pub.publish(msg)
 
    def _publish_state(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        pos_err: np.ndarray,
        vel_err: np.ndarray,
        torque: np.ndarray,
    ):
        state_msg = Float64MultiArray()
        state_msg.data = [
            *q.tolist(),
            *q_dot.tolist(),
            *self.position_reference.tolist(),
            *torque.tolist(),
        ]
        self.controller_state_pub.publish(state_msg)
 
        err_msg = Float64MultiArray()
        err_msg.data = [*pos_err.tolist(), *vel_err.tolist()]
        self.error_pub.publish(err_msg)
 
    def _control_loop(self):
        if not self.controller_active:
            return
 
        if self.last_joint_time is not None:
            age = (
                self.get_clock().now() - self.last_joint_time
            ).nanoseconds * 1e-9
            if age > 0.1:
                self.get_logger().warn(
                    f"Joint state stale ({age:.3f}s). Holding last torque.",
                    throttle_duration_sec=1.0,
                )
                return
 
        q, q_dot = self._get_current_state()
 
        q_ref, q_dot_ref = self.vel_profile.trapezoidal_step(
            q, self.position_reference, self.current_velocity_cmd, self.dt
        )
        q_dot_ref += self.velocity_reference
        self.current_velocity_cmd = q_dot_ref
 
        torque_cmd, pos_err, vel_err = self._cascade_control(
            q, q_dot, q_ref, q_dot_ref
        )
 
        torque_cmd = self._apply_gravity_compensation(torque_cmd, q)
        torque_cmd = self._apply_feedforward(torque_cmd, q_dot)
        torque_cmd = self._saturate_torque(torque_cmd)
 
        self._publish_torque(torque_cmd)
        self._publish_state(q, q_dot, pos_err, vel_err, torque_cmd)
 
        self.prev_vel_cmd = q_dot.copy()
 
    def reset_controllers(self):
        for pid in self.pos_pids + self.vel_pids:
            pid.reset()
        self.current_velocity_cmd = np.zeros(2)
        self.prev_vel_cmd = np.zeros(2)
        self.get_logger().info("Controller state reset.")
 
 
def main(args=None):
    rclpy.init(args=args)
    node = VectorController2DOF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VectorController2DOF.")
    finally:
        torque_msg = Float64MultiArray()
        torque_msg.data = [0.0, 0.0]
        node.torque_pub.publish(torque_msg)
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
