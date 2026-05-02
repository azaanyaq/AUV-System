import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Range, LaserScan, PointCloud2
from std_msgs.msg import Float64MultiArray, Bool, UInt8
from geometry_msgs.msg import Vector3
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
from typing import Optional, List, Dict
import time


class ProximityZone(IntEnum):
    CLEAR = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


class SensorType(IntEnum):
    ULTRASONIC = 0
    INFRARED = 1
    TOF = 2


@dataclass
class SensorConfig:
    name: str
    frame_id: str
    topic: str
    sensor_type: SensorType
    min_range: float
    max_range: float
    field_of_view: float
    mounting_offset: List[float]
    mounting_rpy: List[float]
    enabled: bool = True
    noise_stddev: float = 0.003


@dataclass
class ProximityZoneConfig:
    clear_distance: float = 0.80
    warning_distance: float = 0.45
    critical_distance: float = 0.20
    emergency_distance: float = 0.08


@dataclass
class FilterConfig:
    window_size: int = 12
    outlier_sigma: float = 2.5
    exponential_alpha: float = 0.35
    use_median: bool = True


@dataclass
class ProximitySensorSystemConfig:
    sensors: List[SensorConfig] = field(default_factory=lambda: [
        SensorConfig(
            name="front_ultrasonic",
            frame_id="front_us_link",
            topic="/sensors/front_us/range",
            sensor_type=SensorType.ULTRASONIC,
            min_range=0.02,
            max_range=4.00,
            field_of_view=0.2618,
            mounting_offset=[0.18, 0.0, 0.05],
            mounting_rpy=[0.0, 0.0, 0.0],
        ),
        SensorConfig(
            name="left_infrared",
            frame_id="left_ir_link",
            topic="/sensors/left_ir/range",
            sensor_type=SensorType.INFRARED,
            min_range=0.10,
            max_range=0.80,
            field_of_view=0.0873,
            mounting_offset=[0.10, 0.12, 0.04],
            mounting_rpy=[0.0, 0.0, 1.5708],
        ),
        SensorConfig(
            name="right_infrared",
            frame_id="right_ir_link",
            topic="/sensors/right_ir/range",
            sensor_type=SensorType.INFRARED,
            min_range=0.10,
            max_range=0.80,
            field_of_view=0.0873,
            mounting_offset=[0.10, -0.12, 0.04],
            mounting_rpy=[0.0, 0.0, -1.5708],
        ),
        SensorConfig(
            name="rear_tof",
            frame_id="rear_tof_link",
            topic="/sensors/rear_tof/range",
            sensor_type=SensorType.TOF,
            min_range=0.01,
            max_range=2.00,
            field_of_view=0.4363,
            mounting_offset=[-0.18, 0.0, 0.05],
            mounting_rpy=[0.0, 0.0, 3.1416],
        ),
        SensorConfig(
            name="top_tof",
            frame_id="top_tof_link",
            topic="/sensors/top_tof/range",
            sensor_type=SensorType.TOF,
            min_range=0.01,
            max_range=2.00,
            field_of_view=0.4363,
            mounting_offset=[0.0, 0.0, 0.22],
            mounting_rpy=[0.0, 1.5708, 0.0],
        ),
    ])
    zone_config: ProximityZoneConfig = field(default_factory=ProximityZoneConfig)
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    publish_rate_hz: float = 50.0
    diagnostic_rate_hz: float = 2.0
    sensor_timeout_s: float = 0.5
    velocity_scale_warning: float = 0.65
    velocity_scale_critical: float = 0.25
    velocity_scale_emergency: float = 0.0
    hysteresis_counts: int = 3


class RangeFilter:
    def __init__(self, config: FilterConfig, min_range: float, max_range: float):
        self.config = config
        self.min_range = min_range
        self.max_range = max_range
        self.window: deque = deque(maxlen=config.window_size)
        self.ema_value: Optional[float] = None

    def update(self, raw: float) -> Optional[float]:
        if not (self.min_range <= raw <= self.max_range):
            return self.ema_value

        if len(self.window) >= 3:
            arr = np.array(self.window)
            mean = arr.mean()
            std = arr.std()
            if std > 1e-6 and abs(raw - mean) > self.config.outlier_sigma * std:
                return self.ema_value

        self.window.append(raw)

        if len(self.window) == 0:
            return None

        if self.config.use_median:
            filtered = float(np.median(self.window))
        else:
            filtered = float(np.mean(self.window))

        if self.ema_value is None:
            self.ema_value = filtered
        else:
            self.ema_value = (
                self.config.exponential_alpha * filtered
                + (1.0 - self.config.exponential_alpha) * self.ema_value
            )

        return self.ema_value

    def reset(self):
        self.window.clear()
        self.ema_value = None

    @property
    def ready(self) -> bool:
        return len(self.window) >= max(1, self.config.window_size // 3)


@dataclass
class SensorReading:
    name: str
    raw_distance: float
    filtered_distance: float
    zone: ProximityZone
    timestamp: float
    valid: bool


class ZoneClassifier:
    def __init__(self, zone_config: ProximityZoneConfig, hysteresis_counts: int):
        self.zc = zone_config
        self.hysteresis_counts = hysteresis_counts
        self._current_zones: Dict[str, ProximityZone] = {}
        self._pending_zones: Dict[str, ProximityZone] = {}
        self._pending_counts: Dict[str, int] = {}

    def classify(self, name: str, distance: float) -> ProximityZone:
        if distance <= self.zc.emergency_distance:
            candidate = ProximityZone.EMERGENCY
        elif distance <= self.zc.critical_distance:
            candidate = ProximityZone.CRITICAL
        elif distance <= self.zc.warning_distance:
            candidate = ProximityZone.WARNING
        else:
            candidate = ProximityZone.CLEAR

        current = self._current_zones.get(name, ProximityZone.CLEAR)

        if candidate != current:
            if self._pending_zones.get(name) == candidate:
                self._pending_counts[name] = self._pending_counts.get(name, 0) + 1
                if self._pending_counts[name] >= self.hysteresis_counts:
                    self._current_zones[name] = candidate
                    self._pending_counts[name] = 0
            else:
                self._pending_zones[name] = candidate
                self._pending_counts[name] = 1

            if candidate > current:
                self._current_zones[name] = candidate
                self._pending_counts[name] = 0

        return self._current_zones.get(name, ProximityZone.CLEAR)

    def reset(self, name: str):
        self._current_zones.pop(name, None)
        self._pending_zones.pop(name, None)
        self._pending_counts.pop(name, None)


class VelocityScaler:
    def __init__(self, config: ProximitySensorSystemConfig):
        self.zone_scales = {
            ProximityZone.CLEAR: 1.0,
            ProximityZone.WARNING: config.velocity_scale_warning,
            ProximityZone.CRITICAL: config.velocity_scale_critical,
            ProximityZone.EMERGENCY: config.velocity_scale_emergency,
        }
        self.zone_config = config.zone_config

    def compute_scale(self, distance: float, zone: ProximityZone) -> float:
        if zone == ProximityZone.CLEAR:
            return 1.0
        if zone == ProximityZone.EMERGENCY:
            return 0.0

        if zone == ProximityZone.WARNING:
            d_min = self.zone_config.warning_distance
            d_max = self.zone_config.clear_distance
            scale_min = self.zone_scales[ProximityZone.WARNING]
        else:
            d_min = self.zone_config.critical_distance
            d_max = self.zone_config.warning_distance
            scale_min = self.zone_scales[ProximityZone.CRITICAL]

        t = np.clip((distance - d_min) / (d_max - d_min + 1e-9), 0.0, 1.0)
        return float(scale_min + t * (1.0 - scale_min))

    def aggregate_scale(self, readings: List[SensorReading]) -> float:
        if not readings:
            return 1.0
        scales = [
            self.compute_scale(r.filtered_distance, r.zone)
            for r in readings
            if r.valid
        ]
        if not scales:
            return 1.0
        return float(min(scales))


class ProximitySensorNode(Node):
    def __init__(self):
        super().__init__("proximity_sensor_node")

        self.config = ProximitySensorSystemConfig()

        self.filters: Dict[str, RangeFilter] = {}
        self.latest_readings: Dict[str, SensorReading] = {}
        self.sensor_last_seen: Dict[str, float] = {}

        self.classifier = ZoneClassifier(
            self.config.zone_config, self.config.hysteresis_counts
        )
        self.velocity_scaler = VelocityScaler(self.config)

        self.overall_zone = ProximityZone.CLEAR
        self.estop_active = False

        self._setup_filters()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self._sensor_subscribers = []
        for sensor_cfg in self.config.sensors:
            if not sensor_cfg.enabled:
                continue
            sub = self.create_subscription(
                Range,
                sensor_cfg.topic,
                self._make_range_callback(sensor_cfg),
                sensor_qos,
            )
            self._sensor_subscribers.append(sub)

        self.proximity_pub = self.create_publisher(
            Float64MultiArray,
            "/proximity/distances",
            10,
        )

        self.zone_pub = self.create_publisher(
            UInt8,
            "/proximity/overall_zone",
            reliable_qos,
        )

        self.estop_pub = self.create_publisher(
            Bool,
            "/proximity/emergency_stop",
            reliable_qos,
        )

        self.velocity_scale_pub = self.create_publisher(
            Float64MultiArray,
            "/proximity/velocity_scale",
            10,
        )

        self.zone_vector_pub = self.create_publisher(
            Float64MultiArray,
            "/proximity/zone_vector",
            10,
        )

        self.diagnostic_pub = self.create_publisher(
            DiagnosticArray,
            "/diagnostics",
            10,
        )

        self.publish_timer = self.create_timer(
            1.0 / self.config.publish_rate_hz,
            self._publish_loop,
        )

        self.diagnostic_timer = self.create_timer(
            1.0 / self.config.diagnostic_rate_hz,
            self._publish_diagnostics,
        )

        self.get_logger().info(
            f"ProximitySensorNode online | "
            f"sensors={len(self.config.sensors)} | "
            f"rate={self.config.publish_rate_hz} Hz"
        )

    def _setup_filters(self):
        for sensor_cfg in self.config.sensors:
            self.filters[sensor_cfg.name] = RangeFilter(
                self.config.filter_config,
                sensor_cfg.min_range,
                sensor_cfg.max_range,
            )
            self.latest_readings[sensor_cfg.name] = SensorReading(
                name=sensor_cfg.name,
                raw_distance=float("inf"),
                filtered_distance=float("inf"),
                zone=ProximityZone.CLEAR,
                timestamp=0.0,
                valid=False,
            )

    def _make_range_callback(self, sensor_cfg: SensorConfig):
        def _callback(msg: Range):
            now = time.monotonic()
            self.sensor_last_seen[sensor_cfg.name] = now

            filtered = self.filters[sensor_cfg.name].update(msg.range)
            if filtered is None:
                return

            zone = self.classifier.classify(sensor_cfg.name, filtered)

            self.latest_readings[sensor_cfg.name] = SensorReading(
                name=sensor_cfg.name,
                raw_distance=float(msg.range),
                filtered_distance=filtered,
                zone=zone,
                timestamp=now,
                valid=self.filters[sensor_cfg.name].ready,
            )

        return _callback

    def _check_sensor_timeouts(self) -> List[str]:
        now = time.monotonic()
        timed_out = []
        for sensor_cfg in self.config.sensors:
            if not sensor_cfg.enabled:
                continue
            last = self.sensor_last_seen.get(sensor_cfg.name, 0.0)
            if now - last > self.config.sensor_timeout_s:
                self.latest_readings[sensor_cfg.name].valid = False
                timed_out.append(sensor_cfg.name)
        return timed_out

    def _compute_overall_zone(self, readings: List[SensorReading]) -> ProximityZone:
        valid = [r for r in readings if r.valid]
        if not valid:
            return ProximityZone.CLEAR
        return ProximityZone(max(int(r.zone) for r in valid))

    def _publish_loop(self):
        timed_out = self._check_sensor_timeouts()
        if timed_out:
            self.get_logger().warn(
                f"Sensor timeout: {timed_out}",
                throttle_duration_sec=2.0,
            )

        readings = list(self.latest_readings.values())
        self.overall_zone = self._compute_overall_zone(readings)
        self.estop_active = self.overall_zone == ProximityZone.EMERGENCY

        distance_msg = Float64MultiArray()
        distance_msg.data = [
            r.filtered_distance if r.valid else -1.0 for r in readings
        ]
        self.proximity_pub.publish(distance_msg)

        zone_msg = UInt8()
        zone_msg.data = int(self.overall_zone)
        self.zone_pub.publish(zone_msg)

        estop_msg = Bool()
        estop_msg.data = self.estop_active
        self.estop_pub.publish(estop_msg)

        agg_scale = self.velocity_scaler.aggregate_scale(readings)
        per_sensor_scales = [
            self.velocity_scaler.compute_scale(r.filtered_distance, r.zone)
            if r.valid else 1.0
            for r in readings
        ]
        vel_scale_msg = Float64MultiArray()
        vel_scale_msg.data = [agg_scale] + per_sensor_scales
        self.velocity_scale_pub.publish(vel_scale_msg)

        zone_vector_msg = Float64MultiArray()
        zone_vector_msg.data = [float(r.zone) for r in readings]
        self.zone_vector_pub.publish(zone_vector_msg)

    def _publish_diagnostics(self):
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        now = time.monotonic()

        for sensor_cfg in self.config.sensors:
            status = DiagnosticStatus()
            status.hardware_id = sensor_cfg.name
            status.name = f"proximity/{sensor_cfg.name}"

            reading = self.latest_readings[sensor_cfg.name]
            last_seen = self.sensor_last_seen.get(sensor_cfg.name, 0.0)
            age = now - last_seen if last_seen > 0.0 else float("inf")

            if age > self.config.sensor_timeout_s:
                status.level = DiagnosticStatus.ERROR
                status.message = f"No data for {age:.2f}s"
            elif not reading.valid:
                status.level = DiagnosticStatus.WARN
                status.message = "Filter warming up"
            elif reading.zone == ProximityZone.EMERGENCY:
                status.level = DiagnosticStatus.ERROR
                status.message = "EMERGENCY zone"
            elif reading.zone == ProximityZone.CRITICAL:
                status.level = DiagnosticStatus.WARN
                status.message = "CRITICAL zone"
            else:
                status.level = DiagnosticStatus.OK
                status.message = ProximityZone(reading.zone).name

            status.values = [
                KeyValue(key="filtered_distance_m", value=f"{reading.filtered_distance:.4f}"),
                KeyValue(key="raw_distance_m", value=f"{reading.raw_distance:.4f}"),
                KeyValue(key="zone", value=ProximityZone(reading.zone).name),
                KeyValue(key="data_age_s", value=f"{age:.3f}"),
                KeyValue(key="sensor_type", value=SensorType(sensor_cfg.sensor_type).name),
                KeyValue(key="frame_id", value=sensor_cfg.frame_id),
            ]

            diag_array.status.append(status)

        system_status = DiagnosticStatus()
        system_status.name = "proximity/system"
        system_status.hardware_id = "proximity_sensor_node"
        system_status.values = [
            KeyValue(key="overall_zone", value=ProximityZone(self.overall_zone).name),
            KeyValue(key="estop_active", value=str(self.estop_active)),
            KeyValue(
                key="velocity_scale",
                value=f"{self.velocity_scaler.aggregate_scale(list(self.latest_readings.values())):.3f}",
            ),
            KeyValue(key="active_sensors", value=str(
                sum(1 for r in self.latest_readings.values() if r.valid)
            )),
        ]

        if self.estop_active:
            system_status.level = DiagnosticStatus.ERROR
            system_status.message = "EMERGENCY STOP ACTIVE"
        elif self.overall_zone == ProximityZone.CRITICAL:
            system_status.level = DiagnosticStatus.WARN
            system_status.message = "Critical proximity zone"
        elif self.overall_zone == ProximityZone.WARNING:
            system_status.level = DiagnosticStatus.WARN
            system_status.message = "Warning proximity zone"
        else:
            system_status.level = DiagnosticStatus.OK
            system_status.message = "All clear"

        diag_array.status.append(system_status)
        self.diagnostic_pub.publish(diag_array)

    def get_velocity_scale(self) -> float:
        return self.velocity_scaler.aggregate_scale(
            list(self.latest_readings.values())
        )

    def is_estop(self) -> bool:
        return self.estop_active


def main(args=None):
    rclpy.init(args=args)
    node = ProximitySensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ProximitySensorNode.")
    finally:
        estop_msg = Bool()
        estop_msg.data = False
        node.estop_pub.publish(estop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
