import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Float64MultiArray, Bool, String, UInt8
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import IntEnum, auto
import time
import json


# -------------------------------------------------------------------------
# Mission state machine for the AUV
#
# States in rough order of a typical dive:
#   IDLE → PRE_DIVE_CHECK → DIVING → TRANSIT → STATION_KEEP
#          → SURVEY_LINE → RETURNING → SURFACING → RECOVERY
#
# Emergency paths:
#   Any state → EMERGENCY_SURFACE  (triggered by safety topics)
#   Any state → ABORTED            (triggered by operator or fault)
# -------------------------------------------------------------------------

class MissionState(IntEnum):
    IDLE              = 0
    PRE_DIVE_CHECK    = 1
    DIVING            = 2
    TRANSIT           = 3
    STATION_KEEP      = 4
    SURVEY_LINE       = 5
    RETURNING         = 6
    SURFACING         = 7
    RECOVERY          = 8
    EMERGENCY_SURFACE = 9
    ABORTED           = 10


class WaypointType(IntEnum):
    TRANSIT    = 0
    STATION    = 1   # hover at this point for dwell_s seconds
    SURVEY     = 2   # lawnmower pattern centred on this point
    RETURN     = 3


@dataclass
class Waypoint:
    x: float
    y: float
    z: float   # depth (positive down)
    heading_deg: Optional[float]   # None = free heading
    wp_type: WaypointType  = WaypointType.TRANSIT
    dwell_s: float         = 0.0
    arrival_radius_m: float = 1.5
    label: str             = ""


@dataclass
class MissionConfig:
    # default mission — replace via JSON at runtime
    waypoints: List[Waypoint] = field(default_factory=lambda: [
        Waypoint(x=0.0,   y=0.0,   z=2.0,  heading_deg=0.0,   wp_type=WaypointType.TRANSIT, label="dive point"),
        Waypoint(x=15.0,  y=0.0,   z=5.0,  heading_deg=None,  wp_type=WaypointType.TRANSIT, label="checkpoint alpha"),
        Waypoint(x=30.0,  y=0.0,   z=5.0,  heading_deg=90.0,  wp_type=WaypointType.STATION, dwell_s=20.0, label="survey origin"),
        Waypoint(x=30.0,  y=15.0,  z=5.0,  heading_deg=90.0,  wp_type=WaypointType.SURVEY,  dwell_s=0.0,  label="survey leg 1"),
        Waypoint(x=30.0,  y=-15.0, z=5.0,  heading_deg=270.0, wp_type=WaypointType.SURVEY,  dwell_s=0.0,  label="survey leg 2"),
        Waypoint(x=0.0,   y=0.0,   z=2.0,  heading_deg=180.0, wp_type=WaypointType.RETURN,  label="return to launch"),
    ])
    dive_depth_m:             float = 2.0
    max_mission_duration_s:   float = 1800.0  # 30 min hard limit
    min_battery_pct:          float = 20.0    # abort below this
    transit_speed_mps:        float = 1.2
    station_keep_radius_m:    float = 0.5
    pre_dive_timeout_s:       float = 60.0
    heading_tolerance_deg:    float = 5.0
    update_rate_hz:           float = 10.0
    comms_loss_timeout_s:     float = 120.0   # surface if no topside comms for 2 min


# very rough position error → wrench mapping
# in a real system the nav controller handles this, but we need
# something here to generate demand signals during testing
_TRANSIT_KP_XY  = 8.0
_TRANSIT_KP_Z   = 0.0   # depth controller handles Z
_HEADING_KP     = 12.0
_MAX_FWD_FORCE  = 30.0
_MAX_LATERAL_FORCE = 20.0
_MAX_YAW_TORQUE = 15.0


class PreDiveChecklist:
    """Simple checklist — all items must pass before vehicle dives."""

    def __init__(self):
        self._checks: Dict[str, Optional[bool]] = {
            "imu_healthy":        None,
            "depth_sensor_ok":    None,
            "battery_ok":         None,
            "thrusters_armed":    None,
            "nav_state_valid":    None,
            "proximity_clear":    None,
        }

    def update(self, key: str, passed: bool):
        if key in self._checks:
            self._checks[key] = passed

    def all_passed(self) -> bool:
        return all(v is True for v in self._checks.values())

    def failed_items(self) -> List[str]:
        return [k for k, v in self._checks.items() if v is not True]

    def summary(self) -> str:
        lines = [f"  {'✓' if v else ('?' if v is None else '✗')}  {k}" for k, v in self._checks.items()]
        return "\n".join(lines)


class MissionPlannerNode(Node):

    def __init__(self):
        super().__init__("mission_planner")

        self.cfg         = MissionConfig()
        self.dt          = 1.0 / self.cfg.update_rate_hz

        self.state       = MissionState.IDLE
        self.prev_state  = MissionState.IDLE
        self.wp_index    = 0
        self.dwell_start: Optional[float] = None

        self.position    = np.zeros(3)
        self.heading_deg = 0.0
        self.battery_pct = 100.0
        self.estop_active         = False
        self.proximity_zone       = 0
        self.nav_valid            = False
        self.thrusters_armed      = False
        self.mission_start_time: Optional[float] = None
        self.last_comms_time      = time.monotonic()

        self.checklist = PreDiveChecklist()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pose_sub      = self.create_subscription(PoseStamped,       "/nav/pose",                  self._pose_cb,      sensor_qos)
        self.battery_sub   = self.create_subscription(Float64,           "/battery/state_of_charge",   self._battery_cb,   10)
        self.estop_sub     = self.create_subscription(Bool,              "/safety/emergency_stop",      self._estop_cb,     10)
        self.prox_sub      = self.create_subscription(UInt8,             "/proximity/overall_zone",     self._proximity_cb, sensor_qos)
        self.arm_sub       = self.create_subscription(Bool,              "/thrusters/armed",            self._arm_cb,       10)
        self.start_sub     = self.create_subscription(Bool,              "/mission/start",              self._start_cb,     10)
        self.abort_sub     = self.create_subscription(Bool,              "/mission/abort",              self._abort_cb,     10)
        self.load_sub      = self.create_subscription(String,            "/mission/load_json",          self._load_json_cb, 10)
        self.comms_sub     = self.create_subscription(Bool,              "/comms/topside_heartbeat",    self._comms_cb,     10)

        self.depth_ref_pub    = self.create_publisher(Float64,        "/control/depth_reference",   10)
        self.wrench_pub       = self.create_publisher(WrenchStamped,  "/control/wrench_demand",     sensor_qos)
        self.surface_pub      = self.create_publisher(Bool,           "/mission/command_surface",   10)
        self.state_pub        = self.create_publisher(String,         "/mission/state",             10)
        self.wp_pub           = self.create_publisher(Float64MultiArray, "/mission/current_waypoint", 10)
        self.diag_pub         = self.create_publisher(DiagnosticArray,  "/diagnostics",              10)

        self.main_timer  = self.create_timer(self.dt, self._state_machine)
        self.diag_timer  = self.create_timer(2.0,     self._publish_diag)

        self.get_logger().info("MissionPlanner online — state: IDLE")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _pose_cb(self, msg: PoseStamped):
        self.position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        # crude heading from quaternion — good enough for planner level
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.heading_deg = float(np.degrees(np.arctan2(siny, cosy)))
        self.nav_valid   = True
        self.checklist.update("nav_state_valid", True)

    def _battery_cb(self, msg: Float64):
        self.battery_pct = float(msg.data)
        self.checklist.update("battery_ok", self.battery_pct >= self.cfg.min_battery_pct)

    def _estop_cb(self, msg: Bool):
        self.estop_active = msg.data
        if msg.data and self.state not in (MissionState.EMERGENCY_SURFACE, MissionState.ABORTED):
            self._transition(MissionState.EMERGENCY_SURFACE)

    def _proximity_cb(self, msg: UInt8):
        self.proximity_zone = int(msg.data)
        self.checklist.update("proximity_clear", self.proximity_zone == 0)

    def _arm_cb(self, msg: Bool):
        self.thrusters_armed = msg.data
        self.checklist.update("thrusters_armed", msg.data)

    def _start_cb(self, msg: Bool):
        if not msg.data:
            return
        if self.state == MissionState.IDLE:
            self._transition(MissionState.PRE_DIVE_CHECK)
        else:
            self.get_logger().warn(f"Start received but state is {MissionState(self.state).name}")

    def _abort_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("Abort command received")
            self._transition(MissionState.ABORTED)

    def _load_json_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            wps  = []
            for w in data.get("waypoints", []):
                wps.append(Waypoint(
                    x             = float(w["x"]),
                    y             = float(w["y"]),
                    z             = float(w["z"]),
                    heading_deg   = w.get("heading_deg"),
                    wp_type       = WaypointType(int(w.get("type", 0))),
                    dwell_s       = float(w.get("dwell_s", 0.0)),
                    arrival_radius_m = float(w.get("radius_m", 1.5)),
                    label         = str(w.get("label", "")),
                ))
            self.cfg.waypoints = wps
            self.wp_index      = 0
            self.get_logger().info(f"Loaded mission with {len(wps)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Failed to load mission JSON: {e}")

    def _comms_cb(self, msg: Bool):
        if msg.data:
            self.last_comms_time = time.monotonic()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: MissionState):
        if new_state == self.state:
            return
        self.get_logger().info(
            f"Mission state: {MissionState(self.state).name} → {MissionState(new_state).name}"
        )
        self.prev_state = self.state
        self.state      = new_state
        self.dwell_start = None

        state_msg      = String()
        state_msg.data = MissionState(new_state).name
        self.state_pub.publish(state_msg)

    def _state_machine(self):
        # global guards that can interrupt any state
        if self.mission_start_time is not None:
            elapsed = time.monotonic() - self.mission_start_time
            if elapsed > self.cfg.max_mission_duration_s:
                self.get_logger().warn("Mission time limit reached — surfacing")
                self._transition(MissionState.SURFACING)

        if self.battery_pct < self.cfg.min_battery_pct:
            if self.state not in (MissionState.SURFACING, MissionState.RECOVERY,
                                  MissionState.EMERGENCY_SURFACE, MissionState.ABORTED):
                self.get_logger().warn(f"Low battery ({self.battery_pct:.1f}%) — surfacing")
                self._transition(MissionState.SURFACING)

        comms_age = time.monotonic() - self.last_comms_time
        if comms_age > self.cfg.comms_loss_timeout_s:
            if self.state not in (MissionState.SURFACING, MissionState.RECOVERY,
                                  MissionState.EMERGENCY_SURFACE, MissionState.ABORTED,
                                  MissionState.IDLE):
                self.get_logger().warn(f"Comms loss ({comms_age:.0f}s) — surfacing")
                self._transition(MissionState.SURFACING)

        # dispatch
        {
            MissionState.IDLE:              self._state_idle,
            MissionState.PRE_DIVE_CHECK:    self._state_pre_dive,
            MissionState.DIVING:            self._state_diving,
            MissionState.TRANSIT:           self._state_transit,
            MissionState.STATION_KEEP:      self._state_station_keep,
            MissionState.SURVEY_LINE:       self._state_survey,
            MissionState.RETURNING:         self._state_returning,
            MissionState.SURFACING:         self._state_surfacing,
            MissionState.RECOVERY:          self._state_recovery,
            MissionState.EMERGENCY_SURFACE: self._state_emergency,
            MissionState.ABORTED:           self._state_aborted,
        }.get(self.state, self._state_idle)()

    def _state_idle(self):
        self._publish_zero_wrench()

    def _state_pre_dive(self):
        # checklist updates arrive via subscribers, we just check if done
        self.checklist.update("imu_healthy",     self.nav_valid)
        self.checklist.update("depth_sensor_ok", True)   # TODO: subscribe to depth sensor health

        if self.checklist.all_passed():
            self.get_logger().info("Pre-dive checklist passed:")
            self.get_logger().info(self.checklist.summary())
            self.mission_start_time = time.monotonic()
            self.wp_index = 0
            self._transition(MissionState.DIVING)
        else:
            failed = self.checklist.failed_items()
            self.get_logger().info(
                f"Pre-dive waiting on: {failed}",
                throttle_duration_sec=5.0,
            )

    def _state_diving(self):
        target_depth = self.cfg.dive_depth_m
        self._command_depth(target_depth)
        self._publish_zero_wrench()

        if self.position[2] >= target_depth * 0.85:
            self.get_logger().info(f"Reached dive depth {self.position[2]:.2f} m")
            self._transition(MissionState.TRANSIT)

    def _state_transit(self):
        if self.wp_index >= len(self.cfg.waypoints):
            self.get_logger().info("All waypoints complete — returning")
            self._transition(MissionState.RETURNING)
            return

        wp = self.cfg.waypoints[self.wp_index]
        self._command_depth(wp.z)
        self._command_toward(wp)
        self._publish_current_wp(wp)

        dist = self._distance_to(wp)
        if dist < wp.arrival_radius_m:
            self.get_logger().info(f"Arrived at waypoint {self.wp_index}: '{wp.label}'")
            if wp.wp_type == WaypointType.STATION:
                self._transition(MissionState.STATION_KEEP)
            elif wp.wp_type == WaypointType.SURVEY:
                self._transition(MissionState.SURVEY_LINE)
            elif wp.wp_type == WaypointType.RETURN:
                self._transition(MissionState.RETURNING)
            else:
                self.wp_index += 1

    def _state_station_keep(self):
        wp = self.cfg.waypoints[self.wp_index]
        self._command_depth(wp.z)
        self._command_toward(wp)

        if self.dwell_start is None:
            self.dwell_start = time.monotonic()
            self.get_logger().info(f"Station keep for {wp.dwell_s:.1f}s at '{wp.label}'")

        elapsed = time.monotonic() - self.dwell_start
        if elapsed >= wp.dwell_s:
            self.wp_index += 1
            self._transition(MissionState.TRANSIT)

    def _state_survey(self):
        # simplified — just move to the waypoint and dwell
        # TODO: implement proper lawnmower pattern with line spacing param
        self._state_station_keep()

    def _state_returning(self):
        home = Waypoint(x=0.0, y=0.0, z=self.cfg.dive_depth_m,
                        heading_deg=None, arrival_radius_m=2.0, label="home")
        self._command_depth(home.z)
        self._command_toward(home)
        if self._distance_to(home) < home.arrival_radius_m:
            self._transition(MissionState.SURFACING)

    def _state_surfacing(self):
        self._command_depth(0.1)
        self._publish_zero_wrench()
        cmd = Bool()
        cmd.data = True
        self.surface_pub.publish(cmd)

        if self.position[2] < 0.15:
            self._transition(MissionState.RECOVERY)

    def _state_recovery(self):
        self._publish_zero_wrench()
        self._command_depth(0.0)
        self.get_logger().info("Vehicle recovered — awaiting operator", throttle_duration_sec=30.0)

    def _state_emergency(self):
        self._publish_zero_wrench()
        cmd = Bool()
        cmd.data = True
        self.surface_pub.publish(cmd)
        self.get_logger().error("EMERGENCY SURFACE", throttle_duration_sec=2.0)

    def _state_aborted(self):
        self._publish_zero_wrench()
        cmd = Bool()
        cmd.data = True
        self.surface_pub.publish(cmd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _distance_to(self, wp: Waypoint) -> float:
        return float(np.linalg.norm(self.position - np.array([wp.x, wp.y, wp.z])))

    def _command_depth(self, depth_m: float):
        msg      = Float64()
        msg.data = depth_m
        self.depth_ref_pub.publish(msg)

    def _command_toward(self, wp: Waypoint):
        error_xy = np.array([wp.x - self.position[0], wp.y - self.position[1]])
        dist_xy  = np.linalg.norm(error_xy)

        if dist_xy < 0.5:
            self._publish_zero_wrench()
            return

        direction     = error_xy / (dist_xy + 1e-9)
        desired_hdg   = float(np.degrees(np.arctan2(direction[1], direction[0])))
        hdg_error     = desired_hdg - self.heading_deg
        # normalise to [-180, 180]
        hdg_error     = (hdg_error + 180.0) % 360.0 - 180.0

        surge_force   = np.clip(_TRANSIT_KP_XY * dist_xy, 0.0, _MAX_FWD_FORCE)
        yaw_torque    = np.clip(_HEADING_KP * np.radians(hdg_error), -_MAX_YAW_TORQUE, _MAX_YAW_TORQUE)

        w = WrenchStamped()
        w.header.stamp    = self.get_clock().now().to_msg()
        w.header.frame_id = "base_link"
        w.wrench.force.x  = float(surge_force)
        w.wrench.torque.z = float(yaw_torque)
        self.wrench_pub.publish(w)

    def _publish_zero_wrench(self):
        w = WrenchStamped()
        w.header.stamp    = self.get_clock().now().to_msg()
        w.header.frame_id = "base_link"
        self.wrench_pub.publish(w)

    def _publish_current_wp(self, wp: Waypoint):
        msg      = Float64MultiArray()
        msg.data = [wp.x, wp.y, wp.z, float(self.wp_index)]
        self.wp_pub.publish(msg)

    def _publish_diag(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus()
        s.name        = "mission_planner/state"
        s.hardware_id = "mission_planner"
        is_bad        = self.state in (MissionState.EMERGENCY_SURFACE, MissionState.ABORTED)
        s.level       = DiagnosticStatus.ERROR if is_bad else DiagnosticStatus.OK
        s.message     = MissionState(self.state).name
        elapsed       = time.monotonic() - self.mission_start_time if self.mission_start_time else 0.0
        s.values = [
            KeyValue(key="state",          value=MissionState(self.state).name),
            KeyValue(key="waypoint_index", value=str(self.wp_index)),
            KeyValue(key="total_wps",      value=str(len(self.cfg.waypoints))),
            KeyValue(key="battery_pct",    value=f"{self.battery_pct:.1f}"),
            KeyValue(key="elapsed_s",      value=f"{elapsed:.1f}"),
            KeyValue(key="position_xyz",   value=f"{self.position.tolist()}"),
        ]
        arr.status.append(s)
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
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
