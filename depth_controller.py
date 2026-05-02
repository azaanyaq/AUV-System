import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Float64MultiArray, Bool
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import FluidPressure
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import time


# dedicated depth controller — keep this separate from the vector controller
# buoyancy makes depth a genuinely different problem; coupling it in caused
# instability during the April pool tests so we pulled it out

WATER_DENSITY_KG_M3  = 1025.0   # salt water, roughly correct for the harbour
ATMOSPHERIC_PA       = 101325.0
G                    = 9.81


class DepthMode(IntEnum):
    IDLE         = 0
    DEPTH_HOLD   = 1
    ALTITUDE_HOLD = 2   # hold height above seabed via DVL alt, not implemented yet
    SURFACING    = 3


@dataclass
class DepthPIDGains:
    kp:           float = 55.0
    ki:           float = 1.2
    kd:           float = 14.0
    i_max:        float = 18.0
    d_filter_tau: float = 0.04   # seconds, low-pass on derivative


@dataclass
class DepthControllerConfig:
    gains:                    DepthPIDGains = None
    max_heave_force_N:        float = 50.0
    min_depth_m:              float = 0.15   # don't command shallower than this
    max_depth_m:              float = 80.0
    pressure_topic:           str   = "/sensors/depth_pressure"
    depth_ref_topic:          str   = "/control/depth_reference"
    wrench_out_topic:         str   = "/depth_controller/wrench"
    publish_rate_hz:          float = 50.0
    pressure_timeout_s:       float = 0.5
    surface_threshold_m:      float = 0.10
    buoyancy_trim_N:          float = 2.4    # net buoyancy force (positive = buoyant)
                                             # measured empirically — update after ballasting
    depth_settled_threshold_m: float = 0.03
    depth_settled_count:       int   = 25    # ~0.5s at 50Hz

    def __post_init__(self):
        if self.gains is None:
            self.gains = DepthPIDGains()


class PressureToDepth:
    """
    Converts raw pressure reading to depth in metres.
    Handles tare/offset so we can zero at the surface.
    """

    def __init__(self):
        self._surface_pressure_pa: Optional[float] = None
        self._tare_count  = 0
        self._tare_buffer = []
        self._tare_n      = 20   # average 20 readings for tare

    def tare(self):
        self._tare_buffer = []
        self._tare_count  = 0
        self._surface_pressure_pa = None

    def update_tare(self, pressure_pa: float) -> bool:
        if self._surface_pressure_pa is not None:
            return True
        self._tare_buffer.append(pressure_pa)
        if len(self._tare_buffer) >= self._tare_n:
            self._surface_pressure_pa = float(np.mean(self._tare_buffer))
            return True
        return False

    def to_depth(self, pressure_pa: float) -> Optional[float]:
        if self._surface_pressure_pa is None:
            return None
        delta_pa = pressure_pa - self._surface_pressure_pa
        depth    = delta_pa / (WATER_DENSITY_KG_M3 * G)
        return max(0.0, depth)

    @property
    def tared(self) -> bool:
        return self._surface_pressure_pa is not None


class DepthPID:
    def __init__(self, gains: DepthPIDGains, dt: float):
        self.gains     = gains
        self.dt        = dt
        self.integral  = 0.0
        self.prev_err  = 0.0
        self._deriv_filtered = 0.0
        self._alpha    = dt / (gains.d_filter_tau + dt)
        self._init     = False

    def reset(self):
        self.integral  = 0.0
        self.prev_err  = 0.0
        self._deriv_filtered = 0.0
        self._init     = False

    def compute(self, error: float) -> float:
        if not self._init:
            self.prev_err = error
            self._init    = True
            return self.gains.kp * error

        self.integral += error * self.dt
        self.integral  = np.clip(self.integral, -self.gains.i_max, self.gains.i_max)

        raw_d = (error - self.prev_err) / self.dt
        self._deriv_filtered = (
            self._alpha * raw_d + (1.0 - self._alpha) * self._deriv_filtered
        )
        self.prev_err = error

        return (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * self._deriv_filtered
        )


class DepthControllerNode(Node):

    def __init__(self):
        super().__init__("depth_controller")

        self.cfg  = DepthControllerConfig()
        self.dt   = 1.0 / self.cfg.publish_rate_hz
        self.pid  = DepthPID(self.cfg.gains, self.dt)
        self.p2d  = PressureToDepth()

        self.mode               = DepthMode.IDLE
        self.depth_reference_m  = 0.0
        self.current_depth_m    = 0.0
        self.current_pressure   = 0.0
        self._last_pressure_t   = 0.0
        self._settled_count     = 0
        self._depth_settled     = False
        self._estop             = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pressure_sub  = self.create_subscription(
            FluidPressure,
            self.cfg.pressure_topic,
            self._pressure_cb,
            sensor_qos,
        )
        self.depth_ref_sub = self.create_subscription(
            Float64,
            self.cfg.depth_ref_topic,
            self._depth_ref_cb,
            10,
        )
        self.estop_sub = self.create_subscription(
            Bool,
            "/safety/emergency_stop",
            self._estop_cb,
            10,
        )
        self.surface_cmd_sub = self.create_subscription(
            Bool,
            "/mission/command_surface",
            self._surface_cmd_cb,
            10,
        )

        self.wrench_pub    = self.create_publisher(WrenchStamped, self.cfg.wrench_out_topic, sensor_qos)
        self.depth_pub     = self.create_publisher(Float64,       "/depth_controller/depth_m",     10)
        self.settled_pub   = self.create_publisher(Bool,          "/depth_controller/depth_settled", 10)
        self.diag_pub      = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        self.control_timer = self.create_timer(self.dt,          self._control_loop)
        self.diag_timer    = self.create_timer(2.0,              self._publish_diag)

        # tare automatically on startup — assumes vehicle starts at surface
        self.get_logger().info("DepthController starting — waiting for tare...")

    def _pressure_cb(self, msg: FluidPressure):
        self.current_pressure = msg.fluid_pressure
        self._last_pressure_t = time.monotonic()

        if not self.p2d.tared:
            done = self.p2d.update_tare(msg.fluid_pressure)
            if done:
                self.get_logger().info("Depth sensor tared. DepthController active.")
                self.mode = DepthMode.DEPTH_HOLD
        else:
            d = self.p2d.to_depth(msg.fluid_pressure)
            if d is not None:
                self.current_depth_m = d

    def _depth_ref_cb(self, msg: Float64):
        ref = float(msg.data)
        ref = np.clip(ref, self.cfg.min_depth_m, self.cfg.max_depth_m)
        if abs(ref - self.depth_reference_m) > 0.01:
            self.get_logger().info(f"Depth reference updated: {self.depth_reference_m:.2f} → {ref:.2f} m")
            self._depth_settled = False
            self._settled_count = 0
            self.pid.reset()
        self.depth_reference_m = ref
        self.mode = DepthMode.DEPTH_HOLD

    def _estop_cb(self, msg: Bool):
        self._estop = msg.data

    def _surface_cmd_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("Surface command received — switching to SURFACING mode")
            self.mode = DepthMode.SURFACING
            self.depth_reference_m = 0.0
            self.pid.reset()

    def _control_loop(self):
        if self._estop:
            self._publish_zero_wrench()
            return

        if not self.p2d.tared:
            return

        pressure_age = time.monotonic() - self._last_pressure_t
        if pressure_age > self.cfg.pressure_timeout_s:
            self.get_logger().warn(
                f"Pressure sensor stale ({pressure_age:.2f}s)",
                throttle_duration_sec=2.0,
            )
            self._publish_zero_wrench()
            return

        if self.mode == DepthMode.IDLE:
            self._publish_zero_wrench()
            return

        depth_error = self.depth_reference_m - self.current_depth_m

        # check if we have arrived
        if abs(depth_error) < self.cfg.depth_settled_threshold_m:
            self._settled_count += 1
        else:
            self._settled_count = 0
            self._depth_settled = False

        if self._settled_count >= self.cfg.depth_settled_count:
            if not self._depth_settled:
                self.get_logger().info(f"Depth settled at {self.current_depth_m:.3f} m")
                self._depth_settled = True

        raw_force = self.pid.compute(depth_error)

        # add static buoyancy trim so the integrator doesn't have to do all the work
        # positive trim means vehicle is buoyant, so we need downward force to compensate
        trim_force = -self.cfg.buoyancy_trim_N

        heave_force = np.clip(
            raw_force + trim_force,
            -self.cfg.max_heave_force_N,
            self.cfg.max_heave_force_N,
        )

        if self.mode == DepthMode.SURFACING:
            heave_force = min(heave_force, -self.cfg.max_heave_force_N * 0.5)
            if self.current_depth_m < self.cfg.surface_threshold_m:
                self.get_logger().info("Vehicle at surface — switching to IDLE")
                self.mode = DepthMode.IDLE
                self.pid.reset()

        wrench = WrenchStamped()
        wrench.header.stamp    = self.get_clock().now().to_msg()
        wrench.header.frame_id = "base_link"
        wrench.wrench.force.z  = float(heave_force)

        self.wrench_pub.publish(wrench)

        depth_msg       = Float64()
        depth_msg.data  = self.current_depth_m
        self.depth_pub.publish(depth_msg)

        settled_msg      = Bool()
        settled_msg.data = self._depth_settled
        self.settled_pub.publish(settled_msg)

    def _publish_zero_wrench(self):
        w = WrenchStamped()
        w.header.stamp    = self.get_clock().now().to_msg()
        w.header.frame_id = "base_link"
        self.wrench_pub.publish(w)

    def _publish_diag(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus()
        s.name        = "depth_controller/status"
        s.hardware_id = "depth_controller"
        s.level       = DiagnosticStatus.WARN if not self.p2d.tared else DiagnosticStatus.OK
        s.message     = "Taring..." if not self.p2d.tared else DepthMode(self.mode).name
        s.values = [
            KeyValue(key="depth_m",         value=f"{self.current_depth_m:.4f}"),
            KeyValue(key="reference_m",     value=f"{self.depth_reference_m:.4f}"),
            KeyValue(key="mode",            value=DepthMode(self.mode).name),
            KeyValue(key="settled",         value=str(self._depth_settled)),
            KeyValue(key="pid_integral",    value=f"{self.pid.integral:.4f}"),
            KeyValue(key="buoyancy_trim_N", value=f"{self.cfg.buoyancy_trim_N:.2f}"),
        ]
        arr.status.append(s)
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DepthControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero_wrench()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
