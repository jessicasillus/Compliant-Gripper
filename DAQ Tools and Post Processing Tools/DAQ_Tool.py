#!/usr/bin/env python3
"""
Jessica Sillus
3/31/26

Motor (STS3215) and Sensor Data Acquisition Tool

Reports data from 9 sensors and 2 motors to excel file to be used in post processor tool.
"""

import serial
import time
import os
import threading
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox, Button, Slider
from collections import deque
from tkinter import filedialog, Tk

try:
    from predict_weights import load_calibration, BASELINE_SECONDS
    PREDICT_AVAILABLE = True
except ImportError:
    PREDICT_AVAILABLE = False
    BASELINE_SECONDS  = 3.0

SENSORS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
MIN_UNLOADED_RAW = 500   # baseline

DEFAULT_ESP_PORT   = 'COM4'
DEFAULT_SERVO_PORT = 'COM11'


# Motor Driver
class ST3215Motor:
    INST_READ       = 0x02
    INST_WRITE      = 0x03
    INST_SYNC_WRITE = 0x83

    ADDR_TORQUE_ENABLE      = 0x28
    ADDR_GOAL_POSITION_L    = 0x2A
    ADDR_PRESENT_POSITION_L = 0x38
    ADDR_PRESENT_SPEED_L    = 0x3A
    ADDR_PRESENT_LOAD_L     = 0x40
    ADDR_PRESENT_VOLTAGE    = 0x3E
    ADDR_PRESENT_TEMP       = 0x3F
    ADDR_PRESENT_CURRENT_L  = 0x45

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000, lock=None):
        self._lock = lock if lock is not None else threading.Lock()
        self.serial = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=0.3
        )
        time.sleep(0.1)

    def _checksum(self, data):
        return (~sum(data)) & 0xFF

    def _send_raw(self, motor_id, instruction, params=[]):
        length = len(params) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction] + params
        packet.append(self._checksum(packet[2:]))
        self.serial.write(bytearray(packet))
        time.sleep(0.001)

    def _recv_raw(self, n=8):
        return list(self.serial.read(n))

    def enable_torque(self, motor_id, enable=True):
        with self._lock:
            self._send_raw(motor_id, self.INST_WRITE,
                           [self.ADDR_TORQUE_ENABLE, 1 if enable else 0])

    def set_position(self, motor_id, angle, time_ms=1000):
        position = max(0, min(4095, int((angle / 360.0) * 4095)))
        params = [self.ADDR_GOAL_POSITION_L,
                  position & 0xFF, (position >> 8) & 0xFF,
                  time_ms & 0xFF, (time_ms >> 8) & 0xFF]
        with self._lock:
            self._send_raw(motor_id, self.INST_WRITE, params)

    def sync_write_position(self, motor_positions, time_ms=1000):
        if not motor_positions:
            return
        params = [self.ADDR_GOAL_POSITION_L, 4]
        for motor_id, angle in motor_positions:
            position = max(0, min(4095, int((angle / 360.0) * 4095)))
            tl, th = time_ms & 0xFF, (time_ms >> 8) & 0xFF
            params.extend([motor_id,
                           position & 0xFF, (position >> 8) & 0xFF,
                           tl, th])
        with self._lock:
            self._send_raw(0xFE, self.INST_SYNC_WRITE, params)

    def _read2(self, motor_id, addr):
        with self._lock:
            self.serial.reset_input_buffer()
            self._send_raw(motor_id, self.INST_READ, [addr, 2])
            r = self._recv_raw(8)
        if len(r) >= 7:
            return r[5], r[6]
        return None, None

    def _read1(self, motor_id, addr):
        with self._lock:
            self.serial.reset_input_buffer()
            self._send_raw(motor_id, self.INST_READ, [addr, 1])
            r = self._recv_raw(7)
        if len(r) >= 6:
            return r[5]
        return None

    def read_position(self, motor_id):
        lo, hi = self._read2(motor_id, self.ADDR_PRESENT_POSITION_L)
        if lo is None: return None
        return round(((hi << 8 | lo) / 4095.0) * 360.0, 2)

    def read_speed(self, motor_id):
        lo, hi = self._read2(motor_id, self.ADDR_PRESENT_SPEED_L)
        if lo is None: return None
        raw = (hi << 8) | lo
        return -(raw & 0x3FF) if raw > 1023 else raw

    def read_load(self, motor_id):
        lo, hi = self._read2(motor_id, self.ADDR_PRESENT_LOAD_L)
        if lo is None: return None
        raw = (hi << 8) | lo
        return -(raw & 0x3FF) if raw > 1023 else raw
    
    def read_voltage(self, motor_id):
        v = self._read1(motor_id, self.ADDR_PRESENT_VOLTAGE)
        return round(v / 10.0, 2) if v is not None else None

    def read_temperature(self, motor_id):
        return self._read1(motor_id, self.ADDR_PRESENT_TEMP)

    def read_current(self, motor_id):
        lo, hi = self._read2(motor_id, self.ADDR_PRESENT_CURRENT_L)
        if lo is None: return None
        raw = (hi << 8) | lo
        return raw - 65536 if raw > 32767 else raw

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()

class MotorState:
    M1 = 1
    M2 = 2

    def __init__(self, controller: ST3215Motor):
        self.ctrl   = controller
        self.status = "ready"
        p1 = controller.read_position(self.M1) or 0.0
        p2 = controller.read_position(self.M2) or 0.0
        self.angle = {'m1': p1, 'm2': p2}
        self.home  = {'m1': p1, 'm2': p2}
        self.close_pos = {'m1': None, 'm2': None}
        controller.enable_torque(self.M1, True)
        controller.enable_torque(self.M2, True)

    def _send(self, key, motor_id, angle_deg, time_ms):
        angle_deg = round(max(0.0, min(360.0, angle_deg)), 2)
        self.angle[key] = angle_deg
        self.ctrl.set_position(motor_id, angle_deg, time_ms=time_ms)
        return angle_deg

    def m1_fine(self, delta, time_ms=500):
        a = self._send('m1', self.M1, self.angle['m1'] + delta, time_ms)
        self.status = f"M1 {'+' if delta > 0 else ''}{int(delta)}° → {a:.1f}°"
        print(self.status)

    def m2_fine(self, delta, time_ms=500):
        a = self._send('m2', self.M2, self.angle['m2'] + delta, time_ms)
        self.status = f"M2 {'+' if delta > 0 else ''}{int(delta)}° → {a:.1f}°"
        print(self.status)

    def set_m1_close(self):
        p = self.ctrl.read_position(self.M1)
        if p is not None:
            self.close_pos['m1'] = p
            self.angle['m1']     = p
            self.status = f"M1 close set → {p:.1f}°"
        else:
            self.status = "M1 set closed failed"
        print(self.status)

    def set_m2_close(self):
        p = self.ctrl.read_position(self.M2)
        if p is not None:
            self.close_pos['m2'] = p
            self.angle['m2']     = p
            self.status = f"M2 close set → {p:.1f}°"
        else:
            self.status = "M2 set close failed"
        print(self.status)

    def m1_close(self, time_ms=500):
        if self.close_pos['m1'] is None:
            self.status = "M1 no close setpoint"
            print(self.status); return
        a = self._send('m1', self.M1, self.close_pos['m1'], time_ms)
        self.status = f"M1 → close {a:.1f}°"
        print(self.status)

    def m2_close(self, time_ms=500):
        if self.close_pos['m2'] is None:
            self.status = "M2 no close setpoint"
            print(self.status); return
        a = self._send('m2', self.M2, self.close_pos['m2'], time_ms)
        self.status = f"M2 → close {a:.1f}°"
        print(self.status)

    def home_both(self, time_ms=1000):
        h1, h2 = self.home['m1'], self.home['m2']
        self.angle['m1'] = h1; self.angle['m2'] = h2
        self.ctrl.sync_write_position([(self.M1, h1), (self.M2, h2)], time_ms)
        self.status = f"HOME M1:{h1:.1f}°  M2:{h2:.1f}°"
        print(self.status)

    def set_home(self):
        p1 = self.ctrl.read_position(self.M1)
        p2 = self.ctrl.read_position(self.M2)
        if p1 is not None and p2 is not None:
            self.home['m1'] = p1; self.home['m2'] = p2
            self.angle['m1'] = p1; self.angle['m2'] = p2
            self.status = f"Home set M1:{p1:.1f}°  M2:{p2:.1f}°"
        else:
            self.status = "SET HOME failed"
        print(self.status)

    def shutdown(self):
        self.ctrl.enable_torque(self.M1, False)
        self.ctrl.enable_torque(self.M2, False)
        self.ctrl.close()

class MotorTelemetry:
    MOTOR_COLS = [
        'm1_position_deg', 'm1_load', 'm1_current',
        'm1_speed',        'm1_temp_c', 'm1_voltage_v',
        'm2_position_deg', 'm2_load', 'm2_current',
        'm2_speed',        'm2_temp_c', 'm2_voltage_v',
    ]
    _READS = [
        ('read_position',    'position_deg'),
        ('read_load',        'load'),
        ('read_current',     'current'),
        ('read_speed',       'speed'),
        ('read_temperature', 'temp_c'),
        ('read_voltage',     'voltage_v'),
    ]

    def __init__(self, controller, m1_id=1, m2_id=2, interval=0.3):
        self.ctrl = controller; self.m1_id = m1_id; self.m2_id = m2_id
        self.interval = interval; self.is_running = True
        self._latest = {k: None for k in self.MOTOR_COLS}
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while self.is_running:
            try:
                for method_name, key_suffix in self._READS:
                    fn = getattr(self.ctrl, method_name)
                    v1 = fn(self.m1_id); time.sleep(0.008)
                    v2 = fn(self.m2_id); time.sleep(0.008)
                    self._latest[f'm1_{key_suffix}'] = v1
                    self._latest[f'm2_{key_suffix}'] = v2
            except Exception:
                pass
            time.sleep(self.interval)

    def snapshot(self): return dict(self._latest)
    def stop(self): self.is_running = False

# Calibration file
# Add for live weight prediction on UI

def ask_for_calibration():
    if not PREDICT_AVAILABLE:
        return None, None
    ans = input("Load calibration file for live weight prediction? [y/N]: ").strip().lower()
    if ans != 'y':
        return None, None
    root = Tk(); root.withdraw(); root.attributes("-topmost", True); root.update()
    path = filedialog.askopenfilename(
        parent=root, title="Select calibration_results.xlsx",
        filetypes=[("Excel", "*.xlsx"), ("All", "*.*")])
    root.destroy()
    if not path:
        return None, None
    try:
        calibration, _ = load_calibration(path)
        return calibration, {}
    except Exception:
        return None, None

# Sensor data logger

class SensorLogger:
    def __init__(self, max_points=150, calibration=None):
        self.data_buffer      = []
        self.start_time       = time.time()
        self.is_running       = True
        self.last_note_status = "No notes yet"

        self.max_points   = max_points
        self.times        = deque(maxlen=max_points)
        self.labels       = SENSORS
        self.sensors      = {lbl: deque(maxlen=max_points) for lbl in self.labels}
        self.current_vals = {lbl: 0 for lbl in self.labels}
        self._rolling_window = deque(maxlen=500)

        self.predicted_g   = {lbl: None for lbl in self.labels}
        self.pred_ensemble = None

        # I wired the sensors out of order lol so this is how I had to define them:
        self.raw_map  = ['C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1']
        
        self.calibration      = calibration
        self.baselines        = {}
        self._baselines_ready = False
        self.startup_baselines:  dict = {}
        self.baseline_status:    str  = ""
        self._startup_buf:       list = []
        self._startup_n:         int  = 15

        self.sensitivity_mult   = 1.0
        self.deadband_override  = -1.0

        self.telemetry: MotorTelemetry | None = None

    def _capture_baselines_from_current(self) -> bool:
        new = {}
        for s in self.labels:
            val = self.current_vals.get(s, 0)
            if val >= MIN_UNLOADED_RAW:
                new[s] = float(val)
        if not new:
            return False
        self.baselines = new
        self._baselines_ready = True
        return True

    def _try_init_baselines(self):
        if self._baselines_ready:
            return
        if self.calibration is None:
            return
        self._startup_buf.append(dict(self.current_vals))
        remaining = self._startup_n - len(self._startup_buf)
        if remaining > 0:
            self.baseline_status = f"Capturing startup baseline… ({remaining} packets left)"
            return
        new = {}
        for s in self.labels:
            vals = [row[s] for row in self._startup_buf if row.get(s, 0) >= MIN_UNLOADED_RAW]
            if vals:
                new[s] = float(sum(vals) / len(vals))
        if not new:
            self.baseline_status = "Startup capture failed"
            self._startup_buf.clear()
            return
        self.baselines         = new
        self.startup_baselines = dict(new)
        self._baselines_ready  = True
        self.baseline_status   = (f"Startup baseline ready  "
                                   f"({len(new)} sensors, avg of {self._startup_n} packets)")
        print(f"Startup baselines: "
              f"{ {s: round(v, 1) for s, v in self.startup_baselines.items()} }")

    def set_baseline_now(self):
        if self._capture_baselines_from_current():
            self.baseline_status = "Baseline set now"
            print(f"Baseline set: "
                  f"{ {s: round(v,1) for s,v in self.baselines.items()} }")
            self.predicted_g   = {lbl: None for lbl in self.labels}
            self.pred_ensemble = None
            self._predict_current()
        else:
            self.baseline_status = "Set Baseline failed"
            print(self.baseline_status)

    def revert_to_startup(self):
        if not self.startup_baselines:
            self.baseline_status = "No startup baseline saved yet"
            print(self.baseline_status)
            return
        self.baselines        = dict(self.startup_baselines)
        self._baselines_ready = True
        self.baseline_status  = (
            f"Reverted to startup baseline  "
            f"({ {s: round(v,1) for s,v in self.baselines.items()} })")
        print(f"Reverted to startup baselines: "
              f"{ {s: round(v,1) for s,v in self.baselines.items()} }")
        self.predicted_g   = {lbl: None for lbl in self.labels}
        self.pred_ensemble = None
        self._predict_current()

    def _predict_current(self):
        if self.calibration is None or not self._baselines_ready:
            return
        import numpy as np
        preds = []
        for s in self.labels:
            if s not in self.calibration or s not in self.baselines:
                continue
            cal = self.calibration[s]
            raw_delta = self.baselines[s] - self.current_vals[s]
            deadband = (self.deadband_override
                        if self.deadband_override >= 0
                        else cal["deadband"])
            if abs(raw_delta) < deadband:
                raw_delta = 0.0
            amplified = raw_delta * self.sensitivity_mult
            pred = float(np.clip(
                cal["interp_fn"](np.array([amplified]))[0],
                0.0, float(cal["weight_vals"].max())
            ))
            oor = amplified > cal["oor_threshold"]
            self.predicted_g[s] = (pred, oor)
            preds.append(pred)
        self.pred_ensemble = float(np.median(preds)) if preds else None

    def log_sensors(self, values):
        elapsed = round(time.time() - self.start_time, 3)
        data_point = {'timestamp': elapsed, 'note': ""}
        self.times.append(elapsed)
        for i, val in enumerate(values):
            label = self.raw_map[i]
            data_point[label] = val
            self.sensors[label].append(val)
            self.current_vals[label] = val
        self._rolling_window.append(data_point)
        if self.telemetry is not None:
            data_point.update(self.telemetry.snapshot())
        else:
            for k in MotorTelemetry.MOTOR_COLS:
                data_point[k] = None
        self.data_buffer.append(data_point)
        self._try_init_baselines()
        self._predict_current()

    def log_note(self, note_text):
        if not note_text.strip():
            return
        if self.data_buffer:
            self.data_buffer[-1]['note'] = note_text
            self.last_note_status = f"Last Note: {note_text}"
        else:
            self.last_note_status = f"(buffered) {note_text}"

    def save_excel(self):
        if not self.data_buffer:
            print("No data to save."); return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"sensor_data_{ts}.xlsx")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            df = pd.DataFrame(self.data_buffer)
            cols = ['timestamp'] + self.labels + MotorTelemetry.MOTOR_COLS + ['note']
            df = df.reindex(columns=cols)
            df.to_excel(path, index=False)
            print(f"✓ Saved {len(df)} rows → {path}")
        except Exception as e:
            print(f"Save failed: {e}")


def serial_worker(ser, logger):
    while logger.is_running:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if ',' in line:
                    try:
                        parts  = [x for x in line.split(',') if x.strip()]
                        values = list(map(int, parts))
                        if len(values) == 9:
                            logger.log_sensors(values)
                    except ValueError:
                        pass
        except Exception:
            break
        time.sleep(0.005)

def open_serial_port(default_port, baudrate, timeout, label):

    try:
        ser = serial.Serial(default_port, baudrate, timeout=timeout)
        print(f"✓ {label} connected on {default_port}")
        return ser, default_port
    except Exception:
        print(f"  {label}: could not open {default_port}")
        while True:
            raw = input(f"  Enter {label} port (or blank to skip): ").strip()
            if raw == '':
                return None, None
            try:
                ser = serial.Serial(raw, baudrate, timeout=timeout)
                print(f"✓ {label} connected on {raw}")
                return ser, raw
            except Exception as e:
                print(f"  Failed ({e}) — try again or press Enter to skip.")

def main():
    calibration, _ = ask_for_calibration()
    live_pred = calibration is not None

    # ESP32
    ser, esp_port = open_serial_port(DEFAULT_ESP_PORT, 115200, 0.1, 'ESP32')
    if ser is None:
        print("ESP32 is required")
        return

    motor_state: MotorState | None = None
    telemetry:   MotorTelemetry | None = None

    try:
        shared_lock = threading.Lock()
        ctrl        = ST3215Motor(port=DEFAULT_SERVO_PORT, baudrate=1000000,
                                  lock=shared_lock)
        motor_state = MotorState(ctrl)
        telemetry   = MotorTelemetry(ctrl, interval=0.3)
        print(f"✓ Motors connected on {DEFAULT_SERVO_PORT}")
    except Exception:
        print(f"  Motors: could not open {DEFAULT_SERVO_PORT}")
        raw = input("  Enter servo port (or blank to run without motors): ").strip()
        if raw:
            try:
                shared_lock = threading.Lock()
                ctrl        = ST3215Motor(port=raw, baudrate=1000000, lock=shared_lock)
                motor_state = MotorState(ctrl)
                telemetry   = MotorTelemetry(ctrl, interval=0.3)
                print(f"✓ Motors connected on {raw}")
            except Exception as ex:
                print(f"  Motors unavailable: {ex} continuing without motors.")

    logger = SensorLogger(calibration=calibration)
    logger.telemetry = telemetry
    threading.Thread(target=serial_worker, args=(ser, logger), daemon=True).start()

    # Plots/ Figs
    has_motors    = motor_state is not None
    height_ratios = [8, 2.2, 1.6] if has_motors else [8, 2.2]
    n_rows        = 3 if has_motors else 2

    fig = plt.figure(figsize=(14 if live_pred else 12, 11 if has_motors else 9))
    gs  = fig.add_gridspec(n_rows, 2,
                           width_ratios=[4, 1.6 if live_pred else 1],
                           height_ratios=height_ratios)

    ax_plot = fig.add_subplot(gs[0, 0])
    ax_vals = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, :])
    ax_ctrl.axis('off')

    #sensor
    lines = {lbl: ax_plot.plot([], [], label=lbl)[0] for lbl in logger.labels}
    ax_plot.set_ylim(-100, 4200)
    ax_plot.set_title("Live Sensor Feed & Annotations", fontsize=13)
    ax_plot.legend(loc='upper left', ncol=3, fontsize=11)

    ax_vals.axis('off')
    ax_vals.text(0.0,  0.99, "Sensor",   weight='bold', fontsize=12,
                 transform=ax_vals.transAxes, va='top')
    ax_vals.text(0.35, 0.99, "Raw",      weight='bold', fontsize=12,
                 transform=ax_vals.transAxes, va='top', ha='center')
    if live_pred:
        ax_vals.text(0.78, 0.99, "Pred (g)", weight='bold', fontsize=12,
                     transform=ax_vals.transAxes, va='top', ha='center')

    row_y = [0.91 - i * 0.08 for i in range(len(logger.labels))]
    lbl_texts, raw_texts, pred_texts = {}, {}, {}
    for i, lbl in enumerate(logger.labels):
        y = row_y[i]
        lbl_texts[lbl] = ax_vals.text(0.0,  y, lbl,   fontsize=11,
                                      transform=ax_vals.transAxes, va='top')
        raw_texts[lbl] = ax_vals.text(0.35, y, "---", fontsize=11,
                                      transform=ax_vals.transAxes, va='top', ha='center')
        if live_pred:
            pred_texts[lbl] = ax_vals.text(0.78, y, "---", fontsize=11,
                                           transform=ax_vals.transAxes,
                                           va='top', ha='center')

    ensemble_text = None
    if live_pred:
        ax_vals.plot([0, 1], [0.22, 0.22], color='grey', linewidth=0.8,
                     transform=ax_vals.transAxes, clip_on=False)
        ensemble_text = ax_vals.text(0.0, 0.18, 'Ensemble: ---',
                                     fontsize=12, fontweight='bold',
                                     transform=ax_vals.transAxes, va='top')

    recal_status_text = None
    status_text       = None

    plt.tight_layout(pad=1.5)
    fig.canvas.draw()
    _vbb = ax_vals.get_position()
    _bx  = _vbb.x0 + _vbb.width * 0.05
    _bw  = _vbb.width * 0.9
    _bh  = 0.030

    _update_ref = [None]

    def _do_revert(e):
        logger.revert_to_startup()
        if _update_ref[0] is not None:
            _update_ref[0](None)
        fig.canvas.draw_idle()

    def _do_set_baseline(e):
        logger.set_baseline_now()
        if _update_ref[0] is not None:
            _update_ref[0](None)
        fig.canvas.draw_idle()

    ax_revert_btn = fig.add_axes([_bx, _vbb.y0 + 0.006, _bw, _bh])
    _btn_revert   = Button(ax_revert_btn, 'Revert to Start',
                           color='#d4edff', hovercolor='#a8d8ff')
    _btn_revert.label.set_fontsize(11)
    _btn_revert.on_clicked(_do_revert)

    ax_setbl_btn = fig.add_axes([_bx, _vbb.y0 + 0.006 + _bh + 0.005, _bw, _bh])
    _btn_setbl   = Button(ax_setbl_btn, 'Set Baseline Now',
                          color='#d4f5d4', hovercolor='#a8e8a8')
    _btn_setbl.label.set_fontsize(11)
    _btn_setbl.on_clicked(_do_set_baseline)

    fig.canvas.draw()
    cb  = ax_ctrl.get_position()
    L, B, W, H = cb.x0, cb.y0, cb.width, cb.height

    note_h  = H * 0.38
    note_y  = B + H - note_h
    ax_note = fig.add_axes([L, note_y, W, note_h])

    text_box = TextBox(ax_note, 'Note & Enter: ', initial="", color=".95")
    textbox_focused = [False]

    def on_submit(text):
        logger.log_note(text)
        text_box.set_val("")
        textbox_focused[0] = False

    text_box.on_submit(on_submit)

    def on_click(event):
        textbox_focused[0] = (event.inaxes == ax_note)

    fig.canvas.mpl_connect('button_press_event', on_click)

    slider_h    = H * 0.20
    slider_pad  = W * 0.10
    gap_between = H * 0.04
    slider_w    = W * 0.78

    sens_y  = note_y - gap_between - slider_h
    ax_sens = fig.add_axes([L + slider_pad, sens_y, slider_w, slider_h])
    sl_sens = Slider(ax_sens, 'Sensitivity', 1.0, 5.0,
                     valinit=1.0, valstep=0.1, color='#aad4f5')
    sl_sens.label.set_fontsize(11)
    sl_sens.valtext.set_fontsize(11)

    def on_sens_change(val):
        logger.sensitivity_mult = val

    sl_sens.on_changed(on_sens_change)

    dead_y  = sens_y - gap_between - slider_h
    ax_dead = fig.add_axes([L + slider_pad, dead_y, slider_w, slider_h])
    sl_dead = Slider(ax_dead, 'Deadband', 0.0, 150.0,
                     valinit=-1.0, valstep=1.0, color='#f5d5aa')
    sl_dead.label.set_fontsize(11)
    sl_dead.valtext.set_fontsize(11)

    def _dead_fmt(val):
        return "auto" if val < 0 else f"{int(val)}"

    sl_dead.valtext.set_text(_dead_fmt(sl_dead.val))

    def on_dead_change(val):
        logger.deadband_override = val if val >= 0 else -1.0
        sl_dead.valtext.set_text(_dead_fmt(val))

    sl_dead.on_changed(on_dead_change)

    ax_ctrl.text(
        0.0, 0.16,
        "Sensitivity\n"
        "Deadband",
        transform=ax_ctrl.transAxes, fontsize=9, color='#555555', va='top')

    recal_status_text = ax_ctrl.text(0.0, 0.07, "",
                                     color='darkorange', fontsize=9,
                                     transform=ax_ctrl.transAxes, va='top')
    status_text = ax_ctrl.text(0.0, 0.02, "No notes recorded",
                               color='blue', fontsize=9,
                               transform=ax_ctrl.transAxes, va='top')

    motor_status_text = None
    if has_motors:
        motor_status_text = ax_plot.text(
            0.5, 1.03, "Motor: ready",
            transform=ax_plot.transAxes, ha='center', va='bottom',
            fontsize=11, color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8ffe8', alpha=0.8))

    _motor_btns = {}

    if has_motors:
        fig.canvas.draw()
        ax_bg = fig.add_subplot(gs[2, :])
        ax_bg.axis('off')
        fig.canvas.draw()

        bb   = ax_bg.get_position()
        L2, B2, W2, H2 = bb.x0, bb.y0, bb.width, bb.height
        bh   = min(H2 * 0.55, 0.045)
        by   = B2 + (H2 - bh) / 2

        n         = 10
        group_gap = W2 * 0.018
        side_pad  = W2 * 0.010
        total_gap = side_pad * 2 + group_gap * 2
        bw        = (W2 - total_gap) / n

        xs = []
        x  = L2 + side_pad
        for i in range(n):
            xs.append(x)
            x += bw
            if i in (3, 7):
                x += group_gap

        def add_btn(idx, label, cb, color):
            ax_b = fig.add_axes([xs[idx], by, bw, bh])
            b    = Button(ax_b, label, color=color, hovercolor='#b0d8ff')
            b.on_clicked(cb)
            b.label.set_fontsize(10)
            return b

        _motor_btns['m1_set']   = add_btn(0, 'M1 Set↓',  lambda e: motor_state.set_m1_close(), '#ffe0a0')
        _motor_btns['m1_close'] = add_btn(1, 'M1 Close',  lambda e: motor_state.m1_close(),     '#cce8ff')
        _motor_btns['m1_plus']  = add_btn(2, 'M1 +15°',  lambda e: motor_state.m1_fine(+15),   '#ddeeff')
        _motor_btns['m1_minus'] = add_btn(3, 'M1 -15°',  lambda e: motor_state.m1_fine(-15),   '#ddeeff')
        _motor_btns['m2_set']   = add_btn(4, 'M2 Set↓',  lambda e: motor_state.set_m2_close(), '#ffe0a0')
        _motor_btns['m2_close'] = add_btn(5, 'M2 Close',  lambda e: motor_state.m2_close(),     '#cce8ff')
        _motor_btns['m2_plus']  = add_btn(6, 'M2 +15°',  lambda e: motor_state.m2_fine(+15),   '#ddeeff')
        _motor_btns['m2_minus'] = add_btn(7, 'M2 -15°',  lambda e: motor_state.m2_fine(-15),   '#ddeeff')
        _motor_btns['set_home'] = add_btn(8, 'SET HOME', lambda e: motor_state.set_home(),     '#d4f5d4')
        _motor_btns['home']     = add_btn(9, 'HOME',     lambda e: motor_state.home_both(),    '#d4f5d4')

    matplotlib.rcParams['keymap.save'] = []
    matplotlib.rcParams['keymap.pan']  = []

    def on_key(event):
        if textbox_focused[0]:
            return
        k = (event.key or '').lower()
        if motor_state is None:
            return
        if   k == 'q': motor_state.m1_fine(+15)
        elif k == 'e': motor_state.m1_fine(-15)
        elif k == 'w': motor_state.m1_close()
        elif k == 'a': motor_state.m2_fine(+15)
        elif k == 'd': motor_state.m2_fine(-15)
        elif k == 's': motor_state.m2_close()
        elif k == 'h': motor_state.home_both()
        elif k == 'f': motor_state.set_home()
        else: return
        if motor_status_text is not None:
            motor_status_text.set_text(f"Motor: {motor_state.status}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if not logger.times:
            return list(lines.values())

        curr_times = list(logger.times)
        ax_plot.set_xlim(min(curr_times), max(curr_times) + 1)
        for lbl, line in lines.items():
            line.set_data(curr_times, list(logger.sensors[lbl]))

        for lbl in logger.labels:
            val = logger.current_vals[lbl]
            raw_texts[lbl].set_text(f"{val:4d}")
            raw_texts[lbl].set_color('red' if val < 2000 else 'black')

        if live_pred:
            for lbl in logger.labels:
                info = logger.predicted_g.get(lbl)
                if info is None:
                    pred_texts[lbl].set_text("---")
                    pred_texts[lbl].set_color('grey')
                else:
                    pred_g, oor = info
                    pred_texts[lbl].set_text(f"{pred_g:.0f}")
                    pred_texts[lbl].set_color('darkorange' if oor else 'darkgreen')
            if ensemble_text is not None:
                if logger.pred_ensemble is not None:
                    ensemble_text.set_text(f"Ensemble: {logger.pred_ensemble:.0f} g")
                    ensemble_text.set_color('black')
                else:
                    ensemble_text.set_text("Ensemble: (calibrating…)")
                    ensemble_text.set_color('grey')

        recal_status_text.set_text(logger.baseline_status)
        if motor_status_text is not None and motor_state is not None:
            motor_status_text.set_text(f"Motor: {motor_state.status}")
        status_text.set_text(logger.last_note_status)
        return list(lines.values())

    ani = animation.FuncAnimation(fig, update, interval=50,
                                  blit=False, cache_frame_data=False)
    _update_ref[0] = update

    if not has_motors:
        plt.tight_layout()
    plt.show()

    logger.is_running = False
    if telemetry is not None:
        telemetry.stop()
    logger.save_excel()
    ser.close()
    if motor_state is not None:
        motor_state.shutdown()
    print("✓ Done")


if __name__ == "__main__":
    main()