import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Float64MultiArray, Bool, String, UInt8
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import FluidPressure, Imu, BatteryState
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np
import os
import csv
import json
import time
import struct
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime
from enum import IntEnum


# data logger — writes to CSV and a binary ring buffer
# CSV is human readable for quick review, binary is compact for long dives
#
# file layout:
#   /logs/
#     YYYY-MM-DD_HH-MM-SS/
#       mission_log.csv       ← main log, 10 Hz
#       imu_log.bin           ← binary, 100 Hz (too fast for CSV)
#       events.jsonl          ← one JSON object per line for discrete events
#       metadata.json         ← dive metadata, written on close
#
# ring buffer keeps the last N seconds in memory so we can dump
# pre-event data if something goes wrong


BINARY_IMU_FORMAT  = "<dfffffffffff"   # time, accel xyz, gyro xyz, mag xyz, quat wxyz (partial)
BINARY_IMU_BYTES   = struct.calcsize(BINARY_IMU_FORMAT)

LOG_BASE_DIR = os.path.expanduser("~/auv_logs")


class LoggerState(IntEnum):
    IDLE     = 0
    LOGGING  = 1
    FLUSHING = 2
    ERROR    = 3


@dataclass
class LogEntry:
    timestamp:   float
    depth_m:     float  = 0.0
    depth_ref_m: float  = 0.0
    pos_x:       float  = 0.0
    pos_y:       float  = 0.0
    heading_deg: float  = 0.0
    battery_pct: float  = 0.0
    battery_v:   float  = 0.0
    heave_force: float  = 0.0
    thruster_0:  float  = 0.0
    thruster_1:  float  = 0.0
    thruster_2:  float  = 0.0
    thruster_3:  float  = 0.0
    thruster_4:  float  = 0.0
    thruster_5:  float  = 0.0
    thruster_6:  float  = 0.0
    thruster_7:  float  = 0.0
    prox_zone:   int    = 0
    mission_state: str  = "IDLE"
    estop:       bool   = False

    @staticmethod
    def csv_header() -> List[str]:
        return list(LogEntry.__dataclass_fields__.keys())

    def to_csv_row(self) -> List[Any]:
        return list(asdict(self).values())


@dataclass
class DataLoggerConfig:
    log_rate_hz:           float = 10.0
    imu_rate_hz:           float = 100.0
    ring_buffer_seconds:   float = 30.0
    max_log_size_mb:       float = 512.0
    flush_interval_s:      float = 5.0
    compress_on_close:     bool  = False   # TODO: implement gzip on close
    log_dir:               str   = LOG_BASE_DIR
    auto_start:            bool  = True    # start logging immediately on boot


class BinaryIMUWriter:
    """
    Lightweight binary writer for IMU data.
    Each record is a fixed-size struct so we can mmap and read it back fast.
    """

    def __init__(self, filepath: str):
        self._fp    = open(filepath, "wb")
        self._count = 0
        self._lock  = threading.Lock()

    def write(self, t: float, ax: float, ay: float, az: float,
              gx: float, gy: float, gz: float,
              mx: float, my: float, mz: float,
              qw: float, qx: float, qy: float):
        with self._lock:
            self._fp.write(struct.pack(
                BINARY_IMU_FORMAT, t, ax, ay, az, gx, gy, gz, mx, my, mz, qw, qx, qy
            ))
            self._count += 1

    def flush(self):
        with self._lock:
            self._fp.flush()

    def close(self):
        with self._lock:
            self._fp.close()

    @property
    def record_count(self) -> int:
        return self._count


class DataLoggerNode(Node):

    def __init__(self):
        super().__init__("data_logger")

        self.cfg     = DataLoggerConfig()
        self.state   = LoggerState.IDLE
        self.dt      = 1.0 / self.cfg.log_rate_hz

        self._current_entry = LogEntry(timestamp=time.time())
        self._ring_buffer: deque = deque(
            maxlen=int(self.cfg.ring_buffer_seconds * self.cfg.log_rate_hz)
        )

        self._log_dir:     Optional[str]             = None
        self._csv_file                               = None
        self._csv_writer:  Optional[csv.writer]      = None
        self._imu_writer:  Optional[BinaryIMUWriter] = None
        self._event_file                             = None

        self._session_start:    float = 0.0
        self._rows_written:     int   = 0
        self._last_flush:       float = 0.0
        self._total_bytes:      int   = 0
        self._disk_warning_sent: bool = False
        self._write_lock        = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # subscribe to everything interesting
        self.create_subscription(Float64,         "/depth_controller/depth_m",      self._depth_cb,    sensor_qos)
        self.create_subscription(Float64,         "/control/depth_reference",       self._depth_ref_cb, 10)
        self.create_subscription(PoseStamped,     "/nav/pose",                      self._pose_cb,     sensor_qos)
        self.create_subscription(Float64,         "/battery/state_of_charge",       self._batt_soc_cb, 10)
        self.create_subscription(Float64,         "/battery/voltage",               self._batt_v_cb,   10)
        self.create_subscription(WrenchStamped,   "/depth_controller/wrench",       self._heave_cb,    sensor_qos)
        self.create_subscription(Float64MultiArray, "/thrusters/force_feedback",    self._thruster_cb, sensor_qos)
        self.create_subscription(UInt8,           "/proximity/overall_zone",        self._prox_cb,     sensor_qos)
        self.create_subscription(String,          "/mission/state",                 self._mission_state_cb, 10)
        self.create_subscription(Bool,            "/safety/emergency_stop",         self._estop_cb,    10)
        self.create_subscription(Imu,             "/sensors/imu",                   self._imu_cb,      sensor_qos)

        # control topics
        self.create_subscription(Bool,   "/logging/start", self._start_cb, 10)
        self.create_subscription(Bool,   "/logging/stop",  self._stop_cb,  10)

        self.status_pub = self.create_publisher(String,         "/logging/status",  10)
        self.diag_pub   = self.create_publisher(DiagnosticArray, "/diagnostics",    10)

        self.log_timer  = self.create_timer(self.dt,  self._log_loop)
        self.diag_timer = self.create_timer(5.0,      self._publish_diag)

        if self.cfg.auto_start:
            self._start_session()

        self.get_logger().info(f"DataLogger ready — writing to {self.cfg.log_dir}")

    # ------------------------------------------------------------------
    # Subscriber callbacks — just update the current entry, no heavy work
    # ------------------------------------------------------------------

    def _depth_cb(self, msg: Float64):
        self._current_entry.depth_m = float(msg.data)

    def _depth_ref_cb(self, msg: Float64):
        self._current_entry.depth_ref_m = float(msg.data)

    def _pose_cb(self, msg: PoseStamped):
        self._current_entry.pos_x = msg.pose.position.x
        self._current_entry.pos_y = msg.pose.position.y
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_entry.heading_deg = float(np.degrees(np.arctan2(siny, cosy)))

    def _batt_soc_cb(self, msg: Float64):
        self._current_entry.battery_pct = float(msg.data)

    def _batt_v_cb(self, msg: Float64):
        self._current_entry.battery_v = float(msg.data)

    def _heave_cb(self, msg: WrenchStamped):
        self._current_entry.heave_force = msg.wrench.force.z

    def _thruster_cb(self, msg: Float64MultiArray):
        fields = ["thruster_0","thruster_1","thruster_2","thruster_3",
                  "thruster_4","thruster_5","thruster_6","thruster_7"]
        for i, f in enumerate(fields):
            if i < len(msg.data):
                setattr(self._current_entry, f, float(msg.data[i]))

    def _prox_cb(self, msg: UInt8):
        self._current_entry.prox_zone = int(msg.data)

    def _mission_state_cb(self, msg: String):
        self._current_entry.mission_state = msg.data

    def _estop_cb(self, msg: Bool):
        self._current_entry.estop = msg.data
        if msg.data:
            self._log_event("ESTOP_ACTIVATED", {"timestamp": time.time()})

    def _imu_cb(self, msg: Imu):
        if self._imu_writer is None or self.state != LoggerState.LOGGING:
            return
        a, g, m = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        self._imu_writer.write(
            time.time(),
            a.x, a.y, a.z,
            g.x, g.y, g.z,
            0.0, 0.0, 0.0,   # magnetometer not in Imu msg — blank
            m.w, m.x, m.y,
        )

    def _start_cb(self, msg: Bool):
        if msg.data and self.state != LoggerState.LOGGING:
            self._start_session()

    def _stop_cb(self, msg: Bool):
        if msg.data and self.state == LoggerState.LOGGING:
            self._close_session()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _start_session(self):
        ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(self.cfg.log_dir, ts)
        os.makedirs(self._log_dir, exist_ok=True)

        csv_path   = os.path.join(self._log_dir, "mission_log.csv")
        imu_path   = os.path.join(self._log_dir, "imu_log.bin")
        event_path = os.path.join(self._log_dir, "events.jsonl")

        self._csv_file    = open(csv_path, "w", newline="")
        self._csv_writer  = csv.writer(self._csv_file)
        self._csv_writer.writerow(LogEntry.csv_header())

        self._imu_writer  = BinaryIMUWriter(imu_path)
        self._event_file  = open(event_path, "a")

        self._session_start = time.monotonic()
        self._rows_written  = 0
        self._last_flush    = time.monotonic()
        self._total_bytes   = 0
        self.state          = LoggerState.LOGGING

        self._log_event("SESSION_START", {"dir": self._log_dir, "time": ts})
        self.get_logger().info(f"Logging started → {self._log_dir}")

    def _close_session(self):
        if self.state != LoggerState.LOGGING:
            return

        self.state = LoggerState.FLUSHING
        self.get_logger().info("Closing log session...")

        self._log_event("SESSION_END", {
            "rows":         self._rows_written,
            "imu_records":  self._imu_writer.record_count if self._imu_writer else 0,
            "duration_s":   time.monotonic() - self._session_start,
        })

        metadata = {
            "session_start":  self._session_start,
            "rows_written":   self._rows_written,
            "log_rate_hz":    self.cfg.log_rate_hz,
            "imu_rate_hz":    self.cfg.imu_rate_hz,
            "binary_imu_format": BINARY_IMU_FORMAT,
            "binary_record_bytes": BINARY_IMU_BYTES,
        }
        with open(os.path.join(self._log_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file    = None
            self._csv_writer  = None

        if self._imu_writer:
            self._imu_writer.close()
            self._imu_writer = None

        if self._event_file:
            self._event_file.close()
            self._event_file = None

        self.get_logger().info(
            f"Session closed — {self._rows_written} rows, "
            f"dir={self._log_dir}"
        )
        self.state = LoggerState.IDLE

    def _log_event(self, event_type: str, data: dict):
        if self._event_file is None:
            return
        record = {"type": event_type, "wall_time": time.time(), **data}
        try:
            self._event_file.write(json.dumps(record) + "\n")
            self._event_file.flush()
        except Exception as e:
            self.get_logger().error(f"Event write failed: {e}")

    # ------------------------------------------------------------------
    # Main log loop
    # ------------------------------------------------------------------

    def _log_loop(self):
        now = time.monotonic()

        self._current_entry.timestamp = time.time()
        self._ring_buffer.append(asdict(self._current_entry))

        if self.state != LoggerState.LOGGING:
            return

        if self._csv_writer is None:
            return

        try:
            with self._write_lock:
                self._csv_writer.writerow(self._current_entry.to_csv_row())
                self._rows_written += 1
        except Exception as e:
            self.get_logger().error(f"CSV write error: {e}")
            self.state = LoggerState.ERROR
            return

        if now - self._last_flush > self.cfg.flush_interval_s:
            self._csv_file.flush()
            if self._imu_writer:
                self._imu_writer.flush()
            self._last_flush = now

        self._check_disk_space()

    def _check_disk_space(self):
        try:
            stat = os.statvfs(self._log_dir)
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            if free_mb < 100.0 and not self._disk_warning_sent:
                self.get_logger().warn(f"Low disk space: {free_mb:.0f} MB remaining")
                self._disk_warning_sent = True
            if free_mb < 20.0:
                self.get_logger().error("Critically low disk — stopping logger")
                self._close_session()
        except Exception:
            pass

    def _publish_diag(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus()
        s.name        = "data_logger/status"
        s.hardware_id = "data_logger"
        s.level       = DiagnosticStatus.ERROR if self.state == LoggerState.ERROR else DiagnosticStatus.OK
        s.message     = LoggerState(self.state).name
        elapsed       = time.monotonic() - self._session_start if self._session_start else 0.0
        s.values = [
            KeyValue(key="state",       value=LoggerState(self.state).name),
            KeyValue(key="rows",        value=str(self._rows_written)),
            KeyValue(key="elapsed_s",   value=f"{elapsed:.1f}"),
            KeyValue(key="log_dir",     value=str(self._log_dir or "none")),
            KeyValue(key="ring_buffer", value=f"{len(self._ring_buffer)}/{self._ring_buffer.maxlen}"),
            KeyValue(key="imu_records", value=str(
                self._imu_writer.record_count if self._imu_writer else 0
            )),
        ]
        arr.status.append(s)
        self.diag_pub.publish(arr)

        status_msg      = String()
        status_msg.data = LoggerState(self.state).name
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_session()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
