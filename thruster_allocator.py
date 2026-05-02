import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Wrench, WrenchStamped
from std_msgs.msg import Float64MultiArray, Bool, String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum
import time


# thruster layout is BlueROV2 Heavy config but modified for our 8-thruster frame
# coordinate convention: x=forward, y=left, z=up (NED-ish but z flipped for sanity)
# TODO: double check sign conventions with mech team after next pool test

class ThrusterID(IntEnum):
    FWD_PORT       = 0
    FWD_STBD       = 1
    AFT_PORT       = 2
    AFT_STBD       = 3
    VERT_FWD_PORT  = 4
    VERT_FWD_STBD  = 5
    VERT_AFT_PORT  = 6
    VERT_AFT_STBD  = 7


@dataclass
class ThrusterSpec:
    max_fwd_thrust_N:  float = 35.0   # T500 at 16V
    max_rev_thrust_N:  float = 28.0   # reverse is always weaker
    deadband_pwm:      float = 0.04   # fraction of full scale, empirically tuned
    spin_up_rate:      float = 0.55   # max delta per control cycle (anti-slam)
    pwm_min:           int   = 1100
    pwm_max:           int   = 1900
    pwm_neutral:       int   = 1500


@dataclass
class AllocatorConfig:
    # physical offsets from CoM in metres [x, y, z]
    thruster_positions: List[List[float]] = field(default_factory=lambda: [
        [ 0.18,  0.11,  0.00],   # FWD_PORT
        [ 0.18, -0.11,  0.00],   # FWD_STBD
        [-0.18,  0.11,  0.00],   # AFT_PORT
        [-0.18, -0.11,  0.00],   # AFT_STBD
        [ 0.12,  0.10, -0.06],   # VERT_FWD_PORT
        [ 0.12, -0.10, -0.06],   # VERT_FWD_STBD
        [-0.12,  0.10, -0.06],   # VERT_AFT_PORT
        [-0.12, -0.10, -0.06],   # VERT_AFT_STBD
    ])
    # thrust direction unit vectors for each thruster
    thruster_directions: List[List[float]] = field(default_factory=lambda: [
        [ 1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0,  1.0],
    ])
    max_total_current_A: float  = 80.0
    current_per_newton:  float  = 0.9   # rough estimate, varies with RPM
    publish_rate_hz:     float  = 50.0
    estop_on_timeout:    bool   = True
    wrench_timeout_s:    float  = 0.3


def _build_allocation_matrix(positions, directions) -> np.ndarray:
    # columns are thrusters, rows are [Fx Fy Fz Tx Ty Tz]
    B = np.zeros((6, len(positions)))
    for i, (pos, d) in enumerate(zip(positions, directions)):
        p = np.array(pos)
        d_hat = np.array(d) / (np.linalg.norm(d) + 1e-9)
        torque = np.cross(p, d_hat)
        B[:3, i] = d_hat
        B[3:, i] = torque
    return B


class ThrusterAllocatorNode(Node):

    def __init__(self):
        super().__init__("thruster_allocator")

        self.cfg    = AllocatorConfig()
        self.spec   = ThrusterSpec()
        self.dt     = 1.0 / self.cfg.publish_rate_hz

        self.B      = _build_allocation_matrix(
            self.cfg.thruster_positions,
            self.cfg.thruster_directions,
        )
        self.B_pinv = np.linalg.pinv(self.B)

        self._last_wrench: Optional[np.ndarray] = None
        self._last_wrench_time: float = 0.0
        self._prev_thrust_frac   = np.zeros(8)
        self._estop              = False
        self._killed             = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            "/control/wrench_demand",
            self._wrench_cb,
            sensor_qos,
        )

        self.estop_sub = self.create_subscription(
            Bool,
            "/safety/emergency_stop",
            self._estop_cb,
            10,
        )

        self.kill_sub = self.create_subscription(
            Bool,
            "/safety/kill_switch",
            self._kill_cb,
            10,
        )

        self.pwm_pub   = self.create_publisher(Float64MultiArray, "/thrusters/pwm_commands", sensor_qos)
        self.force_pub = self.create_publisher(Float64MultiArray, "/thrusters/force_feedback", 10)
        self.diag_pub  = self.create_publisher(DiagnosticArray,  "/diagnostics", 10)

        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("ThrusterAllocator ready — 8 thrusters, BlueROV2-Heavy frame")
        self.get_logger().info(f"Allocation matrix condition number: {np.linalg.cond(self.B):.2f}")

    def _wrench_cb(self, msg: WrenchStamped):
        self._last_wrench = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ])
        self._last_wrench_time = time.monotonic()

    def _estop_cb(self, msg: Bool):
        if msg.data and not self._estop:
            self.get_logger().error("E-STOP received — zeroing all thrusters")
        self._estop = msg.data

    def _kill_cb(self, msg: Bool):
        self._killed = msg.data
        if self._killed:
            self.get_logger().fatal("KILL SWITCH ACTIVE")

    def _thrust_to_pwm(self, force_N: float) -> int:
        if force_N >= 0.0:
            frac = min(force_N / self.spec.max_fwd_thrust_N, 1.0)
        else:
            frac = max(force_N / self.spec.max_rev_thrust_N, -1.0)

        if abs(frac) < self.spec.deadband_pwm:
            frac = 0.0

        # linear mapping frac [-1,1] → pwm [1100,1900]
        pwm = int(self.spec.pwm_neutral + frac * (
            (self.spec.pwm_max - self.spec.pwm_neutral) if frac >= 0
            else (self.spec.pwm_neutral - self.spec.pwm_min)
        ))
        return int(np.clip(pwm, self.spec.pwm_min, self.spec.pwm_max))

    def _apply_ramp(self, desired_frac: np.ndarray) -> np.ndarray:
        delta = desired_frac - self._prev_thrust_frac
        delta = np.clip(delta, -self.spec.spin_up_rate, self.spec.spin_up_rate)
        return self._prev_thrust_frac + delta

    def _current_budget_ok(self, thrust_N: np.ndarray) -> bool:
        est_current = np.sum(np.abs(thrust_N)) * self.cfg.current_per_newton
        return est_current <= self.cfg.max_total_current_A

    def _scale_to_budget(self, thrust_N: np.ndarray) -> np.ndarray:
        est_current = np.sum(np.abs(thrust_N)) * self.cfg.current_per_newton
        if est_current <= self.cfg.max_total_current_A:
            return thrust_N
        scale = self.cfg.max_total_current_A / est_current
        self.get_logger().warn(
            f"Current budget exceeded ({est_current:.1f} A), scaling by {scale:.2f}",
            throttle_duration_sec=1.0,
        )
        return thrust_N * scale

    def _control_loop(self):
        # zero out if killed or e-stopped
        if self._killed or self._estop:
            zero_pwm = [self.spec.pwm_neutral] * 8
            msg = Float64MultiArray()
            msg.data = [float(p) for p in zero_pwm]
            self.pwm_pub.publish(msg)
            self._prev_thrust_frac = np.zeros(8)
            return

        # watchdog: zero thrusters if wrench goes stale
        age = time.monotonic() - self._last_wrench_time
        if age > self.cfg.wrench_timeout_s:
            if self._last_wrench is not None:
                self.get_logger().warn(
                    f"Wrench demand stale ({age:.2f}s), holding neutral",
                    throttle_duration_sec=1.0,
                )
            self._prev_thrust_frac = np.zeros(8)
            msg = Float64MultiArray()
            msg.data = [float(self.spec.pwm_neutral)] * 8
            self.pwm_pub.publish(msg)
            return

        wrench = self._last_wrench if self._last_wrench is not None else np.zeros(6)

        # pseudoinverse allocation
        raw_thrust = self.B_pinv @ wrench

        # per-thruster saturation before current check
        for i in range(8):
            if raw_thrust[i] >= 0:
                raw_thrust[i] = min(raw_thrust[i], self.spec.max_fwd_thrust_N)
            else:
                raw_thrust[i] = max(raw_thrust[i], -self.spec.max_rev_thrust_N)

        raw_thrust = self._scale_to_budget(raw_thrust)

        # convert to fraction then ramp-limit
        frac = np.where(
            raw_thrust >= 0,
            raw_thrust / self.spec.max_fwd_thrust_N,
            raw_thrust / self.spec.max_rev_thrust_N,
        )
        frac = self._apply_ramp(frac)
        self._prev_thrust_frac = frac.copy()

        # back to force for feedback topic
        actual_thrust = np.where(
            frac >= 0,
            frac * self.spec.max_fwd_thrust_N,
            frac * self.spec.max_rev_thrust_N,
        )

        pwm_cmds = [self._thrust_to_pwm(float(actual_thrust[i])) for i in range(8)]

        pwm_msg = Float64MultiArray()
        pwm_msg.data = [float(p) for p in pwm_cmds]
        self.pwm_pub.publish(pwm_msg)

        force_msg = Float64MultiArray()
        force_msg.data = actual_thrust.tolist()
        self.force_pub.publish(force_msg)

    def _publish_diagnostics(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus()
        s.name = "thrusters/allocator"
        s.hardware_id = "thruster_allocator"
        s.level = DiagnosticStatus.ERROR if (self._killed or self._estop) else DiagnosticStatus.OK
        s.message = "KILLED" if self._killed else ("E-STOP" if self._estop else "OK")
        s.values = [
            KeyValue(key=f"thruster_{i}_frac", value=f"{self._prev_thrust_frac[i]:.3f}")
            for i in range(8)
        ]
        arr.status.append(s)
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterAllocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero = Float64MultiArray()
        zero.data = [float(node.spec.pwm_neutral)] * 8
        node.pwm_pub.publish(zero)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
