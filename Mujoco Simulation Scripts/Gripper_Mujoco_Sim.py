# Simulate 3-Finger Gripper in MuJoCo with Objects

import time, os, argparse, threading
import numpy as np
import pandas as pd
from datetime import datetime

import mujoco
from mujoco import MjModel, MjData

# Windows DPI fix — must run before any tkinter import
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

SAVE_DIR = r"C:"

EFFECTIVE_RP    = 0.007
H_MOMENT_ARM   = 0.0065
MAX_MOTOR_DEG   = 450.0
CALIB_MAX_DEG   = 700.0
MAX_TENDON_PULL = EFFECTIVE_RP * np.deg2rad(CALIB_MAX_DEG)

E  = 67e6;  bw = 10e-3;  lj = 15e-3
JH = np.array([3e-3, 3.25e-3, 3.25e-3, 4e-3])  # j0–j3 heights
_I = (1.0/12.0) * bw * JH**3
HOLLOW_FACTOR = 0.45   # effective I ≈ 45% of solid cross-section
K_JOINTS = E * _I / lj * HOLLOW_FACTOR

# Force from joint deflection: deficit vs free-air calibration × stiffness
DEFLECTION_GAIN = 300.0   # grams per N·m deflection torque

# Maps raw MuJoCo contact force units → experimental grams
CONTACT_SCALE = 0.14

# Free-air calibration table: int(closure_pct) → np.array([θ_A0..θ_C3])
_free_angle_table = {}

FINGERS   = ['A', 'B', 'C']
GRIPPER_Z = 0.144
DROP_POS  = np.array([0.0, 0.0, GRIPPER_Z + 0.06])

QUAT_CYL_HORIZ = np.array([0.7071068, 0.0, 0.7071068, 0.0])
QUAT_IDENTITY  = np.array([1.0, 0.0, 0.0, 0.0])
QUAT_TAPE_SIDE = np.array([0.7071068, 0.7071068, 0.0, 0.0])

PARK_POS = {
    'obj_paper_roll':  np.array([-0.18,  0.22, 0.0593]),
    'obj_sphere':      np.array([-0.18, -0.22, 0.016 ]),
    'obj_paint_tube':  np.array([ 0.00,  0.22, 0.017 ]),
    'obj_screwdriver': np.array([ 0.18,  0.22, 0.013 ]),
    'obj_tape_roll':   np.array([ 0.00, -0.22, 0.0254]),
}

PARK_QUAT = {
    'obj_paper_roll':  QUAT_CYL_HORIZ,
    'obj_sphere':      QUAT_IDENTITY,
    'obj_paint_tube':  QUAT_CYL_HORIZ,
    'obj_screwdriver': QUAT_CYL_HORIZ,
    'obj_tape_roll':   QUAT_IDENTITY,
}

_jnt_qadr     = {}
_ten_ids      = {}
_act_pos      = {}
_act_frc      = {}
_act_splay    = {}
_splay_qadr   = {}
_obj_jnt_adr  = {}
_obj_jnt_vadr = {}
_touch_sensor = {}
_geom_ids     = {}
_geom_to_seg  = {}

_finger_geom_ids = []
_finger_lip_ids  = []
_finger_jnt_ids  = []
_act_pos_ids     = []

OBJ_NAMES = [
    'obj_paper_roll', 'obj_sphere',
    'obj_paint_tube', 'obj_screwdriver', 'obj_tape_roll',
]

OBJ_DISPLAY = {
    'obj_paper_roll':  'Paper Towel Roll', 'obj_sphere':      'Sphere (Ø32)',
    'obj_paint_tube':  'Paint Tube',       'obj_screwdriver': 'Screwdriver',
    'obj_tape_roll':   'Tape Roll',
}

GRIPPER_Z_UP   = 0.144
GRIPPER_Z_DOWN = 0.304
QUAT_HAND_UP   = np.array([1.0, 0.0, 0.0, 0.0])
QUAT_HAND_DOWN = np.array([0.0, 1.0, 0.0, 0.0])

_orig_geom_data     = {}
_orig_geom_material = {}
_obj_body_ids       = {}

def _store_original_geom_data(model):
    for oname in OBJ_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oname)
        _obj_body_ids[oname] = bid
        glist, mlist = [], []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                glist.append((gid, model.geom_size[gid].copy(), model.geom_pos[gid].copy()))
                mlist.append((gid, model.geom_friction[gid].copy(),
                              model.geom_solref[gid].copy(), model.geom_solimp[gid].copy()))
        _orig_geom_data[oname]     = glist
        _orig_geom_material[oname] = mlist

def apply_object_scale(model, obj_name, scale):
    for gid, orig_size, orig_pos in _orig_geom_data.get(obj_name, []):
        model.geom_size[gid] = orig_size * scale
        model.geom_pos[gid]  = orig_pos * scale

def set_object_weight(model, obj_name, weight_grams):
    bid = _obj_body_ids.get(obj_name)
    if bid is not None:
        model.body_mass[bid] = max(weight_grams / 1000.0, 0.001)

# solref[0] time-constant: rigid metal ≈0.002, wood/plastic ≈0.006, rubber ≈0.015
# solimp[1] dmax: higher → more impedance at contact depth
def set_object_material(model, obj_name, friction=None, solref_tc=None, solimp_dmax=None):
    for gid, _fr, _sr, _si in _orig_geom_material.get(obj_name, []):
        if model.geom_contype[gid] == 0:
            continue
        if friction is not None:
            model.geom_friction[gid, 0] = max(0.1, float(friction))
        if solref_tc is not None:
            model.geom_solref[gid, 0] = float(np.clip(solref_tc, 0.001, 0.05))
        if solimp_dmax is not None:
            model.geom_solimp[gid, 1] = float(np.clip(solimp_dmax, 0.80, 0.999))

def reset_object_material(model, obj_name):
    for gid, orig_fr, orig_sr, orig_si in _orig_geom_material.get(obj_name, []):
        model.geom_friction[gid] = orig_fr.copy()
        model.geom_solref[gid]   = orig_sr.copy()
        model.geom_solimp[gid]   = orig_si.copy()

MATERIAL_PRESETS = {
    'Paint/Rubber': (1.0, 0.016, 0.96),
    'Plastic/Wood': (1.2, 0.007, 0.98),
    'Hard Plastic':  (1.0, 0.004, 0.99),
    'Metal/Rigid':  (0.7, 0.002, 0.999),
}

def _euler_to_quat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg)
    cx, sx = np.cos(rx/2), np.sin(rx/2)
    cy, sy = np.cos(ry/2), np.sin(ry/2)
    cz, sz = np.cos(rz/2), np.sin(rz/2)
    return np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz,
                     cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])

def _quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def set_hand_orientation(model, hand_down):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')
    if hand_down:
        model.body_pos[bid]  = [0, 0, GRIPPER_Z_DOWN]
        model.body_quat[bid] = QUAT_HAND_DOWN
    else:
        model.body_pos[bid]  = [0, 0, GRIPPER_Z_UP]
        model.body_quat[bid] = QUAT_HAND_UP


class ControlPanel:
    FONT_LBL  = ('Segoe UI', 12)
    FONT_BTN  = ('Segoe UI', 11)
    FONT_HDR  = ('Segoe UI', 12, 'bold')
    FONT_SL   = ('Segoe UI', 10)
    FONT_STAT = ('Consolas', 11)

    def __init__(self, shared, lock):
        self.S = shared; self.L = lock; self._root = None

    def _cmd(self, **kw):
        with self.L: self.S.update(kw)

    def _section(self, parent, title):
        from tkinter import ttk
        f = ttk.LabelFrame(parent, text=title, padding=6)
        f.pack(fill='x', padx=8, pady=4)
        return f

    def _btn(self, parent, text, cmd, w=None):
        from tkinter import ttk
        b = ttk.Button(parent, text=text, command=cmd, width=w)
        b.pack(side='left', padx=3, pady=2)
        return b

    def build(self):
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Gripper Control Panel")
        root.geometry("440x1560")
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root = root

        root.option_add('*Font', self.FONT_LBL)
        style = ttk.Style()
        style.configure('.', font=self.FONT_LBL)
        style.configure('TLabelframe.Label', font=self.FONT_HDR)
        style.configure('TLabel', font=self.FONT_LBL)
        style.configure('TButton', font=self.FONT_BTN)
        style.configure('TCombobox', font=self.FONT_LBL)
        style.configure('TCheckbutton', font=self.FONT_LBL)

        canvas = tk.Canvas(root, highlightthickness=0)
        vbar = ttk.Scrollbar(root, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        vbar.pack(side='right', fill='y')
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        fo = self._section(inner, "Orientation")
        r_orient = ttk.Frame(fo); r_orient.pack(fill='x', pady=2)
        self._btn(r_orient, "Hand Up", self._hand_up, 14)
        self._btn(r_orient, "Hand Down", self._hand_down, 14)
        self.orient_lbl = ttk.Label(fo, text="Current: Up",
                                    font=('Segoe UI', 10), foreground='#444')
        self.orient_lbl.pack(anchor='w', padx=4)

        fg = self._section(inner, "Gripper")
        r1 = ttk.Frame(fg); r1.pack(fill='x', pady=2)
        self._btn(r1, "Grasp (Q)", lambda: self._cmd(key='Q'), 14)
        self._btn(r1, "Release (U)", lambda: self._cmd(key='U'), 14)
        r2 = ttk.Frame(fg); r2.pack(fill='x', pady=2)
        self._btn(r2, "Auto-Cycle (Y)", lambda: self._cmd(key='Y'), 14)
        self._btn(r2, "Print Forces (/)", lambda: self._cmd(key='/'), 14)

        rc = ttk.Frame(fg); rc.pack(fill='x', pady=3)
        ttk.Label(rc, text="Closure %").pack(side='left')
        self.closure_var = tk.DoubleVar(value=0)
        self.sl_closure = tk.Scale(rc, from_=0, to=100, orient='horizontal',
                                   variable=self.closure_var, resolution=1,
                                   length=250, font=self.FONT_SL,
                                   command=self._on_closure)
        self.sl_closure.pack(side='left', fill='x', expand=True, padx=4)

        rs = ttk.Frame(fg); rs.pack(fill='x', pady=3)
        ttk.Label(rs, text="Splay deg").pack(side='left')
        self.splay_var = tk.DoubleVar(value=0)
        self.sl_splay = tk.Scale(rs, from_=0, to=30, orient='horizontal',
                                  variable=self.splay_var, resolution=1,
                                  length=250, font=self.FONT_SL,
                                  command=self._on_splay)
        self.sl_splay.pack(side='left', fill='x', expand=True, padx=4)

        rm = ttk.Frame(fg); rm.pack(fill='x', pady=3)
        ttk.Label(rm, text="Max Motor deg").pack(side='left')
        self.max_motor_var = tk.DoubleVar(value=450)
        self.sl_max_motor = tk.Scale(rm, from_=450, to=700, orient='horizontal',
                                      variable=self.max_motor_var, resolution=10,
                                      length=250, font=self.FONT_SL,
                                      command=self._on_max_motor)
        self.sl_max_motor.pack(side='left', fill='x', expand=True, padx=4)

        fob = self._section(inner, "Objects")
        self.obj_var = tk.StringVar(value=OBJ_NAMES[2])
        dl = [OBJ_DISPLAY.get(n, n) for n in OBJ_NAMES]
        self.obj_combo = ttk.Combobox(fob, values=dl, state='readonly',
                                      font=('Segoe UI', 12), width=24)
        self.obj_combo.current(2)
        self.obj_combo.pack(fill='x', padx=2, pady=3)
        self.obj_combo.bind('<<ComboboxSelected>>', self._on_obj_select)

        r8 = ttk.Frame(fob); r8.pack(fill='x', pady=2)
        self._btn(r8, "Drop Selected", self._on_drop, 16)
        self._btn(r8, "Park All (Z)", self._on_park, 14)

        rq = ttk.Frame(fob); rq.pack(fill='x', pady=2)
        self._btn(rq, "Paint",     lambda: self._on_drop_named('obj_paint_tube'),  8)
        self._btn(rq, "Screw",     lambda: self._on_drop_named('obj_screwdriver'), 8)
        self._btn(rq, "Tape Up",   lambda: self._on_drop_named('obj_tape_roll'),   8)
        self._btn(rq, "Tape Side", lambda: self._on_drop_named('obj_tape_roll', tape_side=True), 9)
        rq2 = ttk.Frame(fob); rq2.pack(fill='x', pady=2)
        self._btn(rq2, "Roll",   lambda: self._on_drop_named('obj_paper_roll'), 8)
        self._btn(rq2, "Sphere", lambda: self._on_drop_named('obj_sphere'),     8)

        ft = self._section(inner, "Object Transforms (on Drop)")
        self.rx_var = tk.DoubleVar(value=0)
        self.ry_var = tk.DoubleVar(value=0)
        self.rz_var = tk.DoubleVar(value=0)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.tx_var = tk.DoubleVar(value=0.0)
        self.ty_var = tk.DoubleVar(value=0.0)
        self.tz_var = tk.DoubleVar(value=0.0)
        for label, var, lo, hi, res in [("Rot X", self.rx_var, -180, 180, 5),
                                         ("Rot Y", self.ry_var, -180, 180, 5),
                                         ("Rot Z", self.rz_var, -180, 180, 5),
                                         ("Scale", self.scale_var, 0.2, 3.0, 0.05),
                                         ("Tx (m)", self.tx_var, -0.15, 0.15, 0.005),
                                         ("Ty (m)", self.ty_var, -0.15, 0.15, 0.005),
                                         ("Tz (m)", self.tz_var, -0.10, 0.10, 0.005)]:
            row = ttk.Frame(ft); row.pack(fill='x', pady=1)
            ttk.Label(row, text=label, width=7).pack(side='left')
            tk.Scale(row, from_=lo, to=hi, orient='horizontal', variable=var,
                     resolution=res, length=280, font=self.FONT_SL
                     ).pack(side='left', fill='x', expand=True)

        rw = ttk.Frame(ft); rw.pack(fill='x', pady=3)
        ttk.Label(rw, text="Weight", width=7).pack(side='left')
        self.wt_var = tk.StringVar(value="50.0")
        ttk.Entry(rw, textvariable=self.wt_var, width=8,
                  font=('Segoe UI', 12)).pack(side='left', padx=4)
        ttk.Label(rw, text="g").pack(side='left')
        self._btn(rw, "Set", self._on_weight, 5)

        rr = ttk.Frame(ft); rr.pack(fill='x', pady=2)
        self._btn(rr, "Reset Transforms", self._on_reset_sliders, 18)

        fmat = self._section(inner, "Material (live)")
        self.mat_obj_var = tk.StringVar(value=OBJ_NAMES[2])
        mat_dl = [OBJ_DISPLAY.get(n, n) for n in OBJ_NAMES]
        self.mat_obj_combo = ttk.Combobox(fmat, values=mat_dl, state='readonly',
                                           font=('Segoe UI', 12), width=24)
        self.mat_obj_combo.current(2)
        self.mat_obj_combo.pack(fill='x', padx=2, pady=3)
        self.mat_obj_combo.bind('<<ComboboxSelected>>', self._on_mat_obj_select)

        self.mat_friction_var  = tk.DoubleVar(value=1.1)
        self.mat_stiffness_var = tk.DoubleVar(value=0.008)
        self.mat_dmax_var      = tk.DoubleVar(value=0.98)

        for lbl, var, lo, hi, res, cmd in [
            ("Friction",        self.mat_friction_var,  0.2,   3.0,   0.05,  self._on_mat_friction),
            ("Hardness TC (s)", self.mat_stiffness_var, 0.001, 0.025, 0.001, self._on_mat_stiffness),
            ("Impedance",       self.mat_dmax_var,      0.80,  0.999, 0.005, self._on_mat_dmax),
        ]:
            row = ttk.Frame(fmat); row.pack(fill='x', pady=1)
            ttk.Label(row, text=lbl, width=16, font=self.FONT_SL).pack(side='left')
            tk.Scale(row, from_=lo, to=hi, orient='horizontal', variable=var,
                     resolution=res, length=220, font=self.FONT_SL,
                     command=cmd).pack(side='left', fill='x', expand=True)

        rp1 = ttk.Frame(fmat); rp1.pack(fill='x', pady=2)
        for pname in MATERIAL_PRESETS:
            self._btn(rp1, pname, lambda p=pname: self._on_mat_preset(p), 12)

        rmat_reset = ttk.Frame(fmat); rmat_reset.pack(fill='x', pady=2)
        self._btn(rmat_reset, "Reset Material", self._on_mat_reset, 16)

        ftune = self._section(inner, "Finger Contact (live)")
        self.ft_solref_var    = tk.DoubleVar(value=0.012)
        self.ft_solref_dr_var = tk.DoubleVar(value=1.0)
        self.ft_dmax_var      = tk.DoubleVar(value=0.95)
        self.ft_width_var     = tk.DoubleVar(value=0.008)
        self.ft_damping_var   = tk.DoubleVar(value=0.018)
        self.ft_margin_var    = tk.DoubleVar(value=0.003)
        self.ft_kp_var        = tk.DoubleVar(value=2000)

        for lbl, var, lo, hi, res, cmd in [
            ("Solref TC",   self.ft_solref_var,    0.003, 0.060, 0.001, self._on_ft_solref),
            ("Damp ratio",  self.ft_solref_dr_var, 0.70,  1.10,  0.05,  self._on_ft_solref_dr),
            ("Imped. dmax", self.ft_dmax_var,      0.80,  0.99,  0.01,  self._on_ft_dmax),
            ("Ramp width",  self.ft_width_var,     0.001, 0.030, 0.001, self._on_ft_width),
            ("Joint damp",  self.ft_damping_var,   0.002, 0.080, 0.002, self._on_ft_damping),
            ("Margin",      self.ft_margin_var,    0.000, 0.010, 0.001, self._on_ft_margin),
            ("Actuator Kp", self.ft_kp_var,        500,   8000,  100,   self._on_ft_kp),
        ]:
            row = ttk.Frame(ftune); row.pack(fill='x', pady=1)
            ttk.Label(row, text=lbl, width=14, font=self.FONT_SL).pack(side='left')
            tk.Scale(row, from_=lo, to=hi, orient='horizontal', variable=var,
                     resolution=res, length=210, font=self.FONT_SL,
                     command=cmd).pack(side='left', fill='x', expand=True)

        rft_btns = ttk.Frame(ftune); rft_btns.pack(fill='x', pady=2)
        self._btn(rft_btns, "Stiff",  self._on_ft_preset_stiff, 8)
        self._btn(rft_btns, "Medium", self._on_ft_preset_med, 8)
        self._btn(rft_btns, "Soft",   self._on_ft_preset_soft, 8)

        fsolv = self._section(inner, "Solver (anti-slip)")
        self.sv_noslip_var = tk.IntVar(value=4)
        self.sv_iter_var   = tk.IntVar(value=200)

        for lbl, var, lo, hi, res, cmd in [
            ("Noslip iters", self.sv_noslip_var, 0,  10,  1,  self._on_sv_noslip),
            ("Solver iters", self.sv_iter_var,   50, 500, 10, self._on_sv_iter),
        ]:
            row = ttk.Frame(fsolv); row.pack(fill='x', pady=1)
            ttk.Label(row, text=lbl, width=14, font=self.FONT_SL).pack(side='left')
            tk.Scale(row, from_=lo, to=hi, orient='horizontal', variable=var,
                     resolution=res, length=210, font=self.FONT_SL,
                     command=cmd).pack(side='left', fill='x', expand=True)

        fcal = self._section(inner, "Force Calibration")
        self.fc_cscale_var = tk.DoubleVar(value=CONTACT_SCALE)
        self.fc_dgain_var  = tk.DoubleVar(value=DEFLECTION_GAIN)
        self.fc_gspeed_var = tk.DoubleVar(value=0.4)

        for lbl, var, lo, hi, res, cmd in [
            ("Contact scale",   self.fc_cscale_var, 0.02, 0.60, 0.01, self._on_fc_cscale),
            ("Deflection gain", self.fc_dgain_var,  50,   800,  10,   self._on_fc_dgain),
            ("Grasp speed",     self.fc_gspeed_var, 0.1,  2.0,  0.1,  self._on_fc_gspeed),
        ]:
            row = ttk.Frame(fcal); row.pack(fill='x', pady=1)
            ttk.Label(row, text=lbl, width=16, font=self.FONT_SL).pack(side='left')
            tk.Scale(row, from_=lo, to=hi, orient='horizontal', variable=var,
                     resolution=res, length=210, font=self.FONT_SL,
                     command=cmd).pack(side='left', fill='x', expand=True)

        fs = self._section(inner, "Status")
        self.status_text = tk.Text(fs, height=4, width=46, font=self.FONT_STAT,
                                   bg='#f4f4f4', relief='flat', state='disabled')
        self.status_text.pack(fill='x', padx=4, pady=4)
        self._update_status("Ready")

    def _on_close(self):
        try: self._root.quit(); self._root.destroy()
        except: pass
    def _on_closure(self, _=None):
        self._cmd(set_closure=float(self.closure_var.get()))
    def _on_splay(self, _=None):
        self._cmd(set_splay=float(self.splay_var.get()))
    def _on_max_motor(self, _=None):
        self._cmd(set_max_motor_deg=float(self.max_motor_var.get()))
    def _on_obj_select(self, _=None):
        idx = self.obj_combo.current()
        if 0 <= idx < len(OBJ_NAMES): self.obj_var.set(OBJ_NAMES[idx])
    def _on_drop(self):
        rot   = (self.rx_var.get(), self.ry_var.get(), self.rz_var.get())
        trans = (self.tx_var.get(), self.ty_var.get(), self.tz_var.get())
        self._cmd(drop_obj=self.obj_var.get(), drop_rot=rot,
                  drop_scale=self.scale_var.get(), drop_trans=trans)

    def _on_drop_named(self, obj_name, tape_side=False):
        if obj_name in OBJ_NAMES:
            idx = OBJ_NAMES.index(obj_name)
            self.obj_combo.current(idx)
            self.obj_var.set(obj_name)
        rot   = (self.rx_var.get(), self.ry_var.get(), self.rz_var.get())
        trans = (self.tx_var.get(), self.ty_var.get(), self.tz_var.get())
        self._cmd(drop_obj=obj_name, drop_rot=rot,
                  drop_scale=self.scale_var.get(), drop_trans=trans,
                  drop_tape_side=tape_side)
    def _on_park(self):
        self._cmd(park=True)
    def _hand_up(self):
        self._cmd(hand_down=False)
        self.orient_lbl.config(text="Current: Up")
    def _hand_down(self):
        self._cmd(hand_down=True)
        self.orient_lbl.config(text="Current: Down")
    def _on_weight(self):
        try: g = float(self.wt_var.get())
        except ValueError: self._update_status("Invalid weight"); return
        self._cmd(set_weight=(self.obj_var.get(), g))
        self._update_status(f"Weight → {g:.1f} g for {self.obj_var.get()}")
    def _on_reset_sliders(self):
        self.rx_var.set(0); self.ry_var.set(0); self.rz_var.set(0)
        self.scale_var.set(1.0)
        self.tx_var.set(0.0); self.ty_var.set(0.0); self.tz_var.set(0.0)
        self.splay_var.set(0.0); self._on_splay()

    def _mat_obj(self):
        idx = self.mat_obj_combo.current()
        return OBJ_NAMES[idx] if 0 <= idx < len(OBJ_NAMES) else OBJ_NAMES[4]

    def _on_mat_obj_select(self, _=None):
        idx = self.mat_obj_combo.current()
        if 0 <= idx < len(OBJ_NAMES):
            self.mat_obj_var.set(OBJ_NAMES[idx])

    def _send_mat(self):
        self._cmd(set_material=(self._mat_obj(),
                                float(self.mat_friction_var.get()),
                                float(self.mat_stiffness_var.get()),
                                float(self.mat_dmax_var.get())))

    def _on_mat_friction(self, _=None):   self._send_mat()
    def _on_mat_stiffness(self, _=None):  self._send_mat()
    def _on_mat_dmax(self, _=None):       self._send_mat()

    def _on_mat_preset(self, preset_name):
        fr, tc, dmax = MATERIAL_PRESETS[preset_name]
        self.mat_friction_var.set(fr)
        self.mat_stiffness_var.set(tc)
        self.mat_dmax_var.set(dmax)
        self._cmd(set_material=(self._mat_obj(), fr, tc, dmax))

    def _on_mat_reset(self):
        self._cmd(reset_material=self._mat_obj())

    def _send_finger_contact(self):
        self._cmd(set_finger_contact=(
            float(self.ft_solref_var.get()),
            float(self.ft_dmax_var.get()),
            float(self.ft_width_var.get()),
            float(self.ft_margin_var.get())))
    def _on_ft_solref(self, _=None):    self._send_finger_contact()
    def _on_ft_solref_dr(self, _=None):
        self._cmd(set_finger_solref_dr=float(self.ft_solref_dr_var.get()))
    def _on_ft_dmax(self, _=None):      self._send_finger_contact()
    def _on_ft_width(self, _=None):     self._send_finger_contact()
    def _on_ft_margin(self, _=None):    self._send_finger_contact()
    def _on_ft_damping(self, _=None):
        self._cmd(set_finger_damping=float(self.ft_damping_var.get()))
    def _on_ft_kp(self, _=None):
        self._cmd(set_finger_kp=float(self.ft_kp_var.get()))

    def _on_ft_preset_stiff(self):
        self.ft_solref_var.set(0.012); self.ft_solref_dr_var.set(1.0)
        self.ft_dmax_var.set(0.95)
        self.ft_width_var.set(0.008);  self.ft_damping_var.set(0.018)
        self.ft_margin_var.set(0.003); self.ft_kp_var.set(2000)
        self._send_finger_contact()
        self._on_ft_solref_dr(); self._on_ft_damping(); self._on_ft_kp()
    def _on_ft_preset_med(self):
        self.ft_solref_var.set(0.025); self.ft_solref_dr_var.set(1.0)
        self.ft_dmax_var.set(0.90)
        self.ft_width_var.set(0.012);  self.ft_damping_var.set(0.014)
        self.ft_margin_var.set(0.003); self.ft_kp_var.set(2000)
        self._send_finger_contact()
        self._on_ft_solref_dr(); self._on_ft_damping(); self._on_ft_kp()
    def _on_ft_preset_soft(self):
        self.ft_solref_var.set(0.045); self.ft_solref_dr_var.set(0.90)
        self.ft_dmax_var.set(0.85)
        self.ft_width_var.set(0.018);  self.ft_damping_var.set(0.010)
        self.ft_margin_var.set(0.003); self.ft_kp_var.set(2000)
        self._send_finger_contact()
        self._on_ft_solref_dr(); self._on_ft_damping(); self._on_ft_kp()

    def _on_fc_cscale(self, _=None):
        self._cmd(set_contact_scale=float(self.fc_cscale_var.get()))
    def _on_fc_dgain(self, _=None):
        self._cmd(set_defl_gain=float(self.fc_dgain_var.get()))
    def _on_fc_gspeed(self, _=None):
        self._cmd(set_grasp_speed=float(self.fc_gspeed_var.get()))

    def _on_sv_noslip(self, _=None):
        self._cmd(set_noslip_iter=int(self.sv_noslip_var.get()))
    def _on_sv_iter(self, _=None):
        self._cmd(set_solver_iter=int(self.sv_iter_var.get()))

    def _update_status(self, text):
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', 'end')
        self.status_text.insert('1.0', text)
        self.status_text.configure(state='disabled')

    def refresh_readout(self, closure_pct, mode, tension, selected,
                        grasping, releasing, auto_cycle, hand_down,
                        splay_deg=0.0, max_motor_deg=450.0):
        if not self._root: return
        try:
            self.sl_closure.set(closure_pct)
            self.sl_splay.set(splay_deg)
            self.sl_max_motor.set(max_motor_deg)
            orient = "DOWN" if hand_down else "UP"
            lines = [
                f"Closure: {closure_pct:5.1f}%  Motor: "
                f"{closure_pct/100*max_motor_deg:.0f}°/{max_motor_deg:.0f}°  Hand: {orient}",
                f"Splay: {splay_deg:.0f}°",
            ]
            if grasping:   lines.append(">>> GRASPING ...")
            if releasing:  lines.append(">>> RELEASING ...")
            if auto_cycle: lines.append(">>> AUTO-CYCLE")
            self._update_status('\n'.join(lines))
        except: pass

    def run(self):
        self.build()
        self._root.mainloop()


def _launch_panel(panel):
    try: panel.run()
    except Exception as e: print(f"  [Panel] {e}")


def _build_lookups(model):
    for f in FINGERS:
        for j in range(0, 4):
            name = f'{f}_j{j}'
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            _jnt_qadr[name] = model.jnt_qposadr[jid]

        _ten_ids[f'tendon_{f}'] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TENDON, f'tendon_{f}')
        _act_pos[f] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_pos')
        _act_frc[f] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_frc')

        splay_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_splay')
        if splay_aid >= 0:
            _act_splay[f] = splay_aid
        splay_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{f}_splay')
        if splay_jid >= 0:
            _splay_qadr[f] = model.jnt_qposadr[splay_jid]

        # j0: palm — no physical touch sensor
        for gname in (f'{f}_palm_geom', f'{f}_j0_geom',
                      f'{f}_l0_geom', f'{f}_l0_lip_a', f'{f}_l0_lip_b',
                      f'{f}_l0_lip_c', f'{f}_l0_lip_d'):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                _geom_ids[gname] = gid
                _geom_to_seg[gid] = (f, 0)

        for j in range(1, 4):
            tname = f'{f}_touch{j}'
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, tname)
            if sid >= 0:
                _touch_sensor[tname] = model.sensor_adr[sid]

            gname = f'{f}_l{j}_geom'
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                _geom_ids[gname] = gid
                _geom_to_seg[gid] = (f, j)

            jgname = f'{f}_j{j}_geom'
            jgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, jgname)
            if jgid >= 0:
                _geom_ids[jgname] = jgid
                _geom_to_seg[jgid] = (f, j)

            for suffix in ('lip_a', 'lip_b', 'lip_c', 'lip_d'):
                lgname = f'{f}_l{j}_{suffix}'
                lgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, lgname)
                if lgid >= 0:
                    _geom_ids[lgname] = lgid
                    _geom_to_seg[lgid] = (f, j)

    for oname in OBJ_NAMES:
        jname = oname + '_jnt'
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        _obj_jnt_adr[oname]  = model.jnt_qposadr[jid]
        _obj_jnt_vadr[oname] = model.jnt_dofadr[jid]

    _store_original_geom_data(model)

    _finger_geom_ids.clear()
    _finger_lip_ids.clear()
    _finger_jnt_ids.clear()
    _act_pos_ids.clear()
    for f in FINGERS:
        for j in range(0, 4):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{f}_j{j}')
            if jid >= 0:
                _finger_jnt_ids.append(jid)
            for gname in (f'{f}_l{j}_geom', f'{f}_j{j}_geom'):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    _finger_geom_ids.append(gid)
            for suffix in ('lip_a', 'lip_b', 'lip_c', 'lip_d'):
                lgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{f}_l{j}_{suffix}')
                if lgid >= 0:
                    _finger_lip_ids.append(lgid)
        pgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{f}_palm_geom')
        if pgid >= 0:
            _finger_geom_ids.append(pgid)
        _act_pos_ids.append(_act_pos[f])

    print(f"  Tunable: {len(_finger_geom_ids)} finger geoms, "
          f"{len(_finger_lip_ids)} lip geoms, "
          f"{len(_finger_jnt_ids)} joints, "
          f"{len(_act_pos_ids)} pos actuators")


def set_finger_solref(model, solref_tc):
    # Lip geoms get +0.004 offset for softer initial touch
    tc = float(np.clip(solref_tc, 0.003, 0.060))
    for gid in _finger_geom_ids:
        model.geom_solref[gid, 0] = tc
    for gid in _finger_lip_ids:
        model.geom_solref[gid, 0] = tc + 0.004

def set_finger_solref_damp(model, damp_ratio):
    dr = float(np.clip(damp_ratio, 0.70, 1.10))
    for gid in _finger_geom_ids + _finger_lip_ids:
        model.geom_solref[gid, 1] = dr

def set_finger_solimp(model, dmax=None, width=None):
    for gid in _finger_geom_ids + _finger_lip_ids:
        if dmax is not None:
            model.geom_solimp[gid, 1] = float(np.clip(dmax, 0.80, 0.999))
        if width is not None:
            model.geom_solimp[gid, 2] = float(np.clip(width, 0.001, 0.030))

def set_finger_damping(model, damping):
    d = float(np.clip(damping, 0.002, 0.080))
    for jid in _finger_jnt_ids:
        model.jnt_stiffness  # ensure model is writable
        model.dof_damping[model.jnt_dofadr[jid]] = d

def set_finger_margin(model, margin):
    m = float(np.clip(margin, 0.0, 0.010))
    for gid in _finger_geom_ids + _finger_lip_ids:
        model.geom_margin[gid] = m

def set_actuator_kp(model, kp):
    kp_val = float(np.clip(kp, 500, 8000))
    for aid in _act_pos_ids:
        model.actuator_gainprm[aid, 0] = kp_val
        model.actuator_biasprm[aid, 1] = -kp_val


def calibrate_free_angles(model, data):
    # Sweep 0–100% closure in free air; record joint angles at each 1% step.
    # State is saved and restored — no side effects on the caller.
    global _free_angle_table
    print("  Calibrating free-air joint angles …", end='', flush=True)

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    saved_time = data.time

    for oname in OBJ_NAMES:
        qa = _obj_jnt_adr[oname]
        data.qpos[qa:qa+3] = [5.0, 5.0, -5.0]
        va = _obj_jnt_vadr[oname]
        data.qvel[va:va+6] = 0.0

    settle_steps = 600
    for f in FINGERS:
        if f in _act_splay:
            data.ctrl[_act_splay[f]] = 0.0
    for pct in range(0, 101):
        dphi   = np.deg2rad(pct / 100.0 * CALIB_MAX_DEG)
        target = EFFECTIVE_RP * dphi
        for f in FINGERS:
            data.ctrl[_act_pos[f]] = target
            data.ctrl[_act_frc[f]] = 0.0
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)

        angles = np.zeros(12)
        idx = 0
        for f in FINGERS:
            for j in range(0, 4):
                angles[idx] = data.qpos[_jnt_qadr[f'{f}_j{j}']]
                idx += 1
        _free_angle_table[pct] = angles

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl
    data.time = saved_time
    mujoco.mj_forward(model, data)

    print(f" done ({len(_free_angle_table)} points)")
    for pct in (50, 100):
        a = _free_angle_table[pct]
        degs = np.rad2deg(a)
        print(f"    {pct:3d}%: A=[{degs[0]:.1f},{degs[1]:.1f},{degs[2]:.1f},{degs[3]:.1f}]  "
              f"B=[{degs[4]:.1f},{degs[5]:.1f},{degs[6]:.1f},{degs[7]:.1f}]  "
              f"C=[{degs[8]:.1f},{degs[9]:.1f},{degs[10]:.1f},{degs[11]:.1f}]  (j0–j3)")


def _get_free_angles(closure_pct, max_motor_deg=None):
    # Map closure_pct to calibration scale (built at CALIB_MAX_DEG) then interpolate.
    if max_motor_deg is None:
        max_motor_deg = MAX_MOTOR_DEG
    calib_pct = float(np.clip(closure_pct, 0, 100)) * max_motor_deg / CALIB_MAX_DEG
    calib_pct = float(np.clip(calib_pct, 0, 100))
    lo = int(np.floor(calib_pct))
    hi = min(lo + 1, 100)
    if lo == hi or lo not in _free_angle_table:
        return _free_angle_table.get(round(calib_pct), np.zeros(12))
    frac = calib_pct - lo
    return (1 - frac) * _free_angle_table[lo] + frac * _free_angle_table[hi]


def get_deflection_forces(data, state):
    # Compare actual joint angles to free-air calibration; deficit × stiffness → grams.
    # j0 (palm) has no physical sensor — tracked in sim only.
    # Takes max(deflection_estimate, touch_sensor_estimate) per segment.
    forces_g = {}
    finger_pcts = {}
    for f in FINGERS:
        fc = state['finger_closure'][f]
        finger_pcts[f] = fc if fc is not None else state['closure_pct']

    for f in FINGERS:
        pct  = finger_pcts[f]
        mmd  = state.get('max_motor_deg', MAX_MOTOR_DEG)
        free = _get_free_angles(pct, mmd)
        for j in range(0, 4):
            fi  = FINGERS.index(f)
            idx = fi * 4 + j
            theta_free   = free[idx]
            theta_actual = data.qpos[_jnt_qadr[f'{f}_j{j}']]
            raw_delta = theta_free - theta_actual
            delta     = abs(raw_delta) if abs(raw_delta) > 0.002 else 0.0
            torque    = K_JOINTS[j] * delta
            defl_grams = torque * DEFLECTION_GAIN

            touch_grams = 0.0
            if j > 0:
                tname = f'{f}_touch{j}'
                if tname in _touch_sensor:
                    touch_N     = float(data.sensordata[_touch_sensor[tname]])
                    touch_grams = touch_N / 9.81 * 1000.0 * CONTACT_SCALE

            forces_g[(f, j)] = max(defl_grams, touch_grams)

    return forces_g


def get_finger_angles(data, finger):
    return tuple(data.qpos[_jnt_qadr[f'{finger}_j{j}']] for j in range(0, 4))

def get_tendon_length(data, finger):
    return data.ten_length[_ten_ids[f'tendon_{finger}']]


def get_segment_forces(model, data):
    # Resultant of normal + tangential MuJoCo contact forces, converted to grams.
    # If both geoms are finger segments, split evenly.
    forces = {(f, j): 0.0 for f in FINGERS for j in range(0, 4)}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)
        fn      = abs(wrench[0])
        ft      = np.linalg.norm(wrench[1:3])
        total_g = np.sqrt(fn**2 + ft**2) / 9.81 * 1000.0

        seg1 = _geom_to_seg.get(c.geom1)
        seg2 = _geom_to_seg.get(c.geom2)

        if seg1 and seg2:
            forces[seg1] += total_g * 0.5
            forces[seg2] += total_g * 0.5
        elif seg1:
            forces[seg1] += total_g
        elif seg2:
            forces[seg2] += total_g

    for key in forces:
        forces[key] *= CONTACT_SCALE
    return forces


def get_touch_forces(data):
    forces = {}
    for f in FINGERS:
        for j in range(1, 4):
            tname = f'{f}_touch{j}'
            if tname in _touch_sensor:
                forces[(f, j)] = float(data.sensordata[_touch_sensor[tname]])
            else:
                forces[(f, j)] = 0.0
    return forces


def print_segment_forces(model, data, state=None):
    # L0* = palm/j0, no physical sensor
    sf     = get_segment_forces(model, data)
    defl_g = get_deflection_forces(data, state) if state else {}

    print(f"\n{'─'*80}")
    print(f"  PER-SEGMENT FORCES  @ t = {data.time:.2f}s")
    print(f"  (L0*=palm/j0 no sensor | Contact=MuJoCo solver grams | Defl=angle-based grams)")
    print(f"{'─'*80}")
    print(f"  {'Finger':>8s}  {'Seg':>5s}  {'Contact(g)':>12s}  {'Defl(g)':>10s}  {'Δθ(°)':>9s}")
    print(f"  {'─'*58}")

    for f in FINGERS:
        total_cnt = total_defl = 0.0
        for j in range(0, 4):
            cg = sf.get((f, j), 0.0)
            dg = defl_g.get((f, j), 0.0)
            total_cnt  += cg
            total_defl += dg
            if state and _free_angle_table:
                pct = state['finger_closure'][f]
                if pct is None: pct = state['closure_pct']
                mmd  = state.get('max_motor_deg', MAX_MOTOR_DEG)
                free = _get_free_angles(pct, mmd)
                fi   = FINGERS.index(f)
                theta_free = free[fi * 4 + j]
                theta_act  = data.qpos[_jnt_qadr[f'{f}_j{j}']]
                delta_deg  = np.rad2deg(theta_free - theta_act)
            else:
                delta_deg = 0.0
            label = f'L{j}*' if j == 0 else f'L{j} '
            print(f"  {f:>8s}  {label:>5s}  {cg:12.1f}  {dg:10.1f}  {delta_deg:+9.2f}")
        print(f"  {f+' tot':>8s}  {'':>5s}  {total_cnt:12.1f}  {total_defl:10.1f}")
        print(f"  {'─'*58}")

    grand_cnt  = sum(v for k,v in sf.items() if k[1]>0)
    grand_defl = sum(v for k,v in defl_g.items() if k[1]>0)
    print(f"  {'SENSED':>8s}  {'':>5s}  {grand_cnt:12.1f}  {grand_defl:10.1f}")
    print(f"{'─'*80}\n")


class SimLogger:
    # Logs sim data in a format comparable to experimental sensor_data Excel files.
    LOG_INTERVAL = 0.05   # seconds between rows (~20 Hz)

    def __init__(self):
        self.rows = []
        self.start_time = None
        self.last_log_time = -1.0

    def log(self, model, data, state):
        if self.start_time is None:
            self.start_time = data.time
        if data.time - self.last_log_time < self.LOG_INTERVAL:
            return
        self.last_log_time = data.time

        elapsed    = round(data.time - self.start_time, 3)
        seg_forces = get_segment_forces(model, data)
        defl_g     = get_deflection_forces(data, state)

        motor_deg     = state['closure_pct'] / 100.0 * state.get('max_motor_deg', MAX_MOTOR_DEG)
        dphi          = np.deg2rad(motor_deg)
        tendon_target = EFFECTIVE_RP * dphi

        row = {'timestamp': elapsed}

        for f in FINGERS:
            row[f'{f}0_sim'] = round(seg_forces.get((f, 0), 0.0), 2)
            for j in range(1, 4):
                row[f'{f}{j}'] = round(seg_forces.get((f, j), 0.0), 2)

        row['m1_position_deg'] = None
        row['m1_load']         = None
        row['m1_current']      = None
        row['m2_position_deg'] = round(motor_deg, 2)
        row['m2_load']         = None
        row['m2_current']      = None

        for f in FINGERS:
            th0, th1, th2, th3 = get_finger_angles(data, f)
            tlen = get_tendon_length(data, f)
            row[f'th0_{f}'] = round(np.rad2deg(th0), 3)
            row[f'th1_{f}'] = round(np.rad2deg(th1), 3)
            row[f'th2_{f}'] = round(np.rad2deg(th2), 3)
            row[f'th3_{f}'] = round(np.rad2deg(th3), 3)

            sum_th = th0 + th1 + th2 + th3
            Fc = K_JOINTS.mean() * sum_th / H_MOMENT_ARM if sum_th > 0 else 0.0
            row[f'Fc_{f}'] = round(Fc, 5)

            row[f'{f}0_defl_g'] = round(defl_g.get((f, 0), 0.0), 2)
            for j in range(1, 4):
                row[f'{f}{j}_defl_g'] = round(defl_g.get((f, j), 0.0), 2)

        row['tendon_target_mm'] = round(tendon_target * 1000, 4)
        row['closure_pct']      = round(state['closure_pct'], 1)
        row['max_motor_deg']    = round(state.get('max_motor_deg', MAX_MOTOR_DEG), 1)
        row['splay_deg']        = round(state.get('splay_deg', 0.0), 1)
        row['contact_scale']    = round(CONTACT_SCALE, 4)
        row['deflection_gain']  = round(DEFLECTION_GAIN, 1)
        row['note']             = ''
        self.rows.append(row)

    def save_excel(self):
        if not self.rows:
            print('  No simulation data to save.')
            return
        ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'mujoco_sim_{ts}'
        print(f'\n  Default filename: {default_name}.xlsx')
        print(f'  Save directory  : {SAVE_DIR}')
        try:
            user_input = input('  Enter filename (no extension), or Enter for default '
                               '["skip" = don\'t save]: ').strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ''
        if user_input.lower() == 'skip':
            print('  Save skipped.')
            return
        name = user_input if user_input else default_name
        if name.lower().endswith('.xlsx'):
            name = name[:-5]
        path = os.path.join(SAVE_DIR, f'{name}.xlsx')
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            pd.DataFrame(self.rows).to_excel(path, index=False)
            print(f'  Saved {len(self.rows)} rows -> {path}')
        except Exception as e:
            print(f'  Save failed: {e}')
            fallback = os.path.join(SAVE_DIR, f'{default_name}.xlsx')
            try:
                pd.DataFrame(self.rows).to_excel(fallback, index=False)
                print(f'  Fallback saved -> {fallback}')
            except Exception as e2:
                print(f'  Fallback also failed: {e2}')


def _set_object_pose(data, obj_name, pos, quat):
    qa = _obj_jnt_adr[obj_name]
    va = _obj_jnt_vadr[obj_name]
    data.qpos[qa:qa+3]   = pos
    data.qpos[qa+3:qa+7] = quat
    data.qvel[va:va+6]   = 0.0

def drop_object(data, obj_name, tape_side=False,
                rotation_deg=None, model=None, scale=None, hand_down=False,
                translation_m=None):
    if scale is not None and model is not None and scale != 1.0:
        apply_object_scale(model, obj_name, scale)

    base_z = GRIPPER_Z_DOWN if hand_down else GRIPPER_Z
    if obj_name == 'obj_paper_roll':
        dz = -0.14 if hand_down else 0.14
        drop_pos  = np.array([0.0, 0.0, base_z + dz])
        base_quat = QUAT_CYL_HORIZ.copy()
    elif obj_name == 'obj_tape_roll':
        dz = -0.08 if hand_down else 0.08
        drop_pos  = np.array([0.0, 0.0, base_z + dz])
        base_quat = QUAT_TAPE_SIDE.copy() if tape_side else QUAT_IDENTITY.copy()
    elif obj_name == 'obj_sphere':
        dz = -0.06 if hand_down else 0.06
        drop_pos  = np.array([0.0, 0.0, base_z + dz])
        base_quat = QUAT_IDENTITY.copy()
    else:
        dz = -0.06 if hand_down else 0.06
        drop_pos  = np.array([0.0, 0.0, base_z + dz])
        base_quat = QUAT_CYL_HORIZ.copy()

    if rotation_deg is not None:
        rx, ry, rz = rotation_deg
        if abs(rx) > 0.1 or abs(ry) > 0.1 or abs(rz) > 0.1:
            eq = _euler_to_quat(rx, ry, rz)
            base_quat = _quat_mul(eq, base_quat)
            n = np.linalg.norm(base_quat)
            if n > 1e-8: base_quat /= n

    if translation_m is not None:
        tx, ty, tz = translation_m
        drop_pos = drop_pos + np.array([tx, ty, tz])

    _set_object_pose(data, obj_name, drop_pos, base_quat)


def park_all_objects(data, model=None):
    for oname in OBJ_NAMES:
        if model is not None:
            apply_object_scale(model, oname, 1.0)
        _set_object_pose(data, oname, PARK_POS[oname].copy(), PARK_QUAT[oname])


def print_detailed(state, model, data):
    mmd    = state.get('max_motor_deg', MAX_MOTOR_DEG)
    dphi   = np.deg2rad(state['closure_pct'] / 100.0 * mmd)
    target = EFFECTIVE_RP * dphi
    inv_k  = 1.0 / K_JOINTS
    total_theta = EFFECTIVE_RP * dphi / H_MOMENT_ARM if dphi > 0 else 0
    a_thetas    = (inv_k / inv_k.sum()) * total_theta if dphi > 0 else np.zeros(4)
    seg_forces  = get_segment_forces(model, data)
    defl_g      = get_deflection_forces(data, state)

    print(f"\n{'═'*80}")
    print(f"  DETAILED STATE @ t = {data.time:.3f}s  (defl_gain={DEFLECTION_GAIN:.0f})")
    print(f"{'═'*80}")
    print(f"  Closure: {state['closure_pct']:.1f}%   Motor: "
          f"{state['closure_pct']/100*mmd:.1f}°"
          f"   Mode: {'POS' if state['mode']=='pos' else 'FRC'}"
          f"   Tension: {state['tension']:.1f} N")
    print(f"  Tendon target: {target*1e3:.4f} mm   "
          f"(hollow={HOLLOW_FACTOR:.0%}, k0={K_JOINTS[0]:.4f} k1={K_JOINTS[1]:.4f} "
          f"k2={K_JOINTS[2]:.4f} k3={K_JOINTS[3]:.4f})")
    print(f"  Analytical (no-load j0..j3): [{np.rad2deg(a_thetas[0]):.2f}°, "
          f"{np.rad2deg(a_thetas[1]):.2f}°, {np.rad2deg(a_thetas[2]):.2f}°, "
          f"{np.rad2deg(a_thetas[3]):.2f}°]  Σ={np.rad2deg(total_theta):.1f}°")
    print(f"{'─'*80}")
    print(f"  {'Finger':>8s}  {'θ0*':>8s}  {'θ1':>8s}  {'θ2':>8s}  {'θ3':>8s}  "
          f"{'Σθ':>8s}  {'Tendon':>10s}  (* = palm/j0, no sensor)")
    for f in FINGERS:
        th0, th1, th2, th3 = get_finger_angles(data, f)
        tlen = get_tendon_length(data, f)
        d = [np.rad2deg(x) for x in (th0, th1, th2, th3)]
        print(f"  {f:>8s}  {d[0]:7.2f}°*{d[1]:8.2f}° {d[2]:8.2f}° {d[3]:8.2f}°  "
              f"{sum(d):8.1f}°  {tlen*1e3:8.3f} mm")
        for j in range(0, 4):
            dg    = defl_g.get((f, j), 0.0)
            cnt_g = seg_forces.get((f, j), 0.0)
            pct   = state['finger_closure'][f]
            if pct is None: pct = state['closure_pct']
            free       = _get_free_angles(pct, mmd)
            fi         = FINGERS.index(f)
            theta_free = free[fi * 4 + j]
            theta_act  = data.qpos[_jnt_qadr[f'{f}_j{j}']]
            delta_deg  = np.rad2deg(theta_free - theta_act)
            if dg > 0.1 or cnt_g > 0.1:
                tag = '*' if j == 0 else ' '
                print(f"  {'':>8s}    L{j}{tag}: contact={cnt_g:6.1f}g  "
                      f"defl={dg:6.1f}g  Δθ={delta_deg:+.2f}°")
        h_sum = H_MOMENT_ARM * (th0 + th1 + th2 + th3)
        print(f"  {'':>8s}  h·Σθ = {h_sum*1e3:.4f} mm  "
              f"(target {target*1e3:.4f}  err {(h_sum-target)*1e3:+.4f} mm)")
    print(f"{'═'*80}\n")


def main():
    global CONTACT_SCALE, DEFLECTION_GAIN
    parser = argparse.ArgumentParser(description='3-Finger Gripper — Interactive MuJoCo Simulation v9d')
    parser.add_argument('--closure', type=float, default=0.0)
    parser.add_argument('--mode', choices=['pos', 'frc'], default='pos')
    parser.add_argument('--tension', type=float, default=2.0)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--no-panel', action='store_true', help='Disable tkinter control panel')
    args = parser.parse_args()

    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_3finger.xml")
    model = MjModel.from_xml_path(xml_path)
    data  = MjData(model)
    _build_lookups(model)
    calibrate_free_angles(model, data)

    state = {
        'closure_pct':    np.clip(args.closure, 0, 100),
        'mode':           args.mode,
        'tension':        args.tension,
        'auto_cycle':     args.auto,
        'direction':      1,
        'speed':          0.3,
        'selected':       None,
        'finger_closure': {'A': None, 'B': None, 'C': None},
        'grasping':       False,
        'releasing':      False,
        'grasp_speed':    0.4,
        'hand_down':      False,
        'max_motor_deg':  MAX_MOTOR_DEG,
        'splay_deg':      0.0,
    }

    n_substeps = 20
    logger     = SimLogger()
    panel_lock = threading.Lock()
    panel_cmds = {}
    panel_obj  = None

    def apply_controls():
        mmd = state['max_motor_deg']
        for f in FINGERS:
            fc     = state['finger_closure'][f]
            pct    = fc if fc is not None else state['closure_pct']
            target = EFFECTIVE_RP * np.deg2rad(pct / 100.0 * mmd)
            if state['mode'] == 'pos':
                data.ctrl[_act_pos[f]] = target
                data.ctrl[_act_frc[f]] = 0.0
            else:
                data.ctrl[_act_pos[f]] = 0.0
                data.ctrl[_act_frc[f]] = state['tension']

        splay_rad = np.deg2rad(state['splay_deg'])
        if 'A' in _act_splay: data.ctrl[_act_splay['A']] = splay_rad
        if 'B' in _act_splay: data.ctrl[_act_splay['B']] = splay_rad

    def reset():
        mujoco.mj_resetData(model, data)
        state['closure_pct'] = 0.0;  state['direction'] = 1
        state['grasping'] = state['releasing'] = False
        state['finger_closure'] = {'A': None, 'B': None, 'C': None}
        state['tension']   = args.tension
        state['splay_deg'] = 0.0
        data.ctrl[:] = 0
        set_hand_orientation(model, state['hand_down'])
        mujoco.mj_forward(model, data)
        park_all_objects(data, model)
        print("  >> RESET")

    def handle_key(k):
        sel = state['selected']
        if k in ('Q','q'):
            state['grasping']=True; state['releasing']=state['auto_cycle']=False
            print(f"  >> GRASP ({state['max_motor_deg']:.0f}°)"); return True
        if k in ('U','u'):
            state['releasing']=True; state['grasping']=state['auto_cycle']=False
            print("  >> RELEASE"); return True
        if k in ('Y','y'):
            state['auto_cycle']=not state['auto_cycle']; state['grasping']=state['releasing']=False
            print(f"  >> Cycle: {'ON' if state['auto_cycle'] else 'OFF'}"); return True
        if k==']':
            state['grasping']=state['releasing']=False
            if sel:
                c=state['finger_closure'][sel]; c=c if c is not None else state['closure_pct']
                state['finger_closure'][sel]=min(100,c+5)
            else: state['closure_pct']=min(100,state['closure_pct']+5); state['auto_cycle']=False
            return True
        if k=='[':
            state['grasping']=state['releasing']=False
            if sel:
                c=state['finger_closure'][sel]; c=c if c is not None else state['closure_pct']
                state['finger_closure'][sel]=max(0,c-5)
            else: state['closure_pct']=max(0,state['closure_pct']-5); state['auto_cycle']=False
            return True
        if k in ('=','+'):
            state['grasping']=state['releasing']=False
            if sel: c=state['finger_closure'][sel] or state['closure_pct']; state['finger_closure'][sel]=min(100,c+1)
            else: state['closure_pct']=min(100,state['closure_pct']+1); state['auto_cycle']=False
            return True
        if k=='-':
            state['grasping']=state['releasing']=False
            if sel: c=state['finger_closure'][sel] or state['closure_pct']; state['finger_closure'][sel]=max(0,c-1)
            else: state['closure_pct']=max(0,state['closure_pct']-1); state['auto_cycle']=False
            return True
        if k=='9':
            state['mode']='frc' if state['mode']=='pos' else 'pos'
            print(f"  >> {'POSITION' if state['mode']=='pos' else 'FORCE'}"); return True
        if k=='.': state['tension']=min(10,state['tension']+0.5); return True
        if k==',': state['tension']=max(0,state['tension']-0.5); return True
        if k=='5': state['selected']='A'; print("  >> Finger A"); return True
        if k=='6': state['selected']='B'; print("  >> Finger B"); return True
        if k=='7': state['selected']='C'; print("  >> Finger C"); return True
        if k=='8':
            state['selected']=None; state['finger_closure']={'A':None,'B':None,'C':None}
            print("  >> ALL"); return True
        hd = state['hand_down']
        if k in ('R','r'): drop_object(data,'obj_paper_roll',hand_down=hd); print("  >> ROLL"); return True
        if k==';': drop_object(data,'obj_sphere',hand_down=hd); print("  >> SPHERE"); return True
        if k in ('V','v'): drop_object(data,'obj_paint_tube',hand_down=hd); print("  >> PAINT TUBE"); return True
        if k in ('D','d'): drop_object(data,'obj_screwdriver',hand_down=hd); print("  >> SCREWDRIVER"); return True
        if k in ('J','j'): drop_object(data,'obj_tape_roll',tape_side=False,hand_down=hd); print("  >> TAPE↑"); return True
        if k in ('H','h'): drop_object(data,'obj_tape_roll',tape_side=True,hand_down=hd); print("  >> TAPE→"); return True
        if k in ('Z','z','\\'):
            park_all_objects(data,model); print("  >> Parked"); return True
        if k=='/': print_segment_forces(model, data, state); return True
        if k=='`': print_detailed(state, model, data); return True
        return False

    def key_cb(keycode):
        if keycode==59: handle_key(';')
        elif keycode==92: handle_key('\\')
        elif keycode==96: handle_key('`')
        else:
            try: handle_key(chr(keycode))
            except: pass

    reset()
    print("\n" + "=" * 60)
    print("  3-FINGER GRIPPER — Interactive Sim v9d")
    print("    condim=6 + center lips + noslip solver")
    print("=" * 60)
    print(f"  rp={EFFECTIVE_RP*1e3:.1f}mm  h={H_MOMENT_ARM*1e3:.1f}mm  "
          f"hollow={HOLLOW_FACTOR:.0%}  max={state['max_motor_deg']:.0f}°")
    print(f"  solref=0.012  noslip=4  condim=6  lip=2mm×4/link")
    print(f"  Splay: 0–30°  |  Closure: 450–700°  |  friction=2.8")
    print("=" * 60)

    if not args.no_panel:
        try:
            panel_obj = ControlPanel(panel_cmds, panel_lock)
            threading.Thread(target=_launch_panel, args=(panel_obj,), daemon=True).start()
            print("  ★ Control Panel launched")
        except Exception as e:
            print(f"  Panel failed: {e}"); panel_obj = None
    print()

    import mujoco.viewer as viewer
    with viewer.launch_passive(model, data, key_callback=key_cb) as v:
        v.cam.lookat[:] = [0, 0, 0.144]
        v.cam.distance  = 0.55
        v.cam.elevation = -25
        v.cam.azimuth   = 145

        last_print = last_panel = 0.0

        while v.is_running():
            t0 = time.time()

            with panel_lock:
                cmds = dict(panel_cmds); panel_cmds.clear()

            with v.lock():
                if 'key' in cmds: handle_key(cmds['key'])
                if 'set_closure' in cmds:
                    state['closure_pct']=float(np.clip(cmds['set_closure'],0,100))
                    state['grasping']=state['releasing']=state['auto_cycle']=False
                if 'set_tension' in cmds:
                    state['tension']=float(np.clip(cmds['set_tension'],0,10))
                if 'hand_down' in cmds:
                    hd = bool(cmds['hand_down'])
                    state['hand_down'] = hd
                    set_hand_orientation(model, hd)
                    mujoco.mj_forward(model, data)
                    orient = "DOWN" if hd else "UP"
                    print(f"  >> Hand {orient}")
                    if hd:
                        v.cam.lookat[:] = [0, 0, GRIPPER_Z_DOWN]
                        v.cam.elevation = 25
                    else:
                        v.cam.lookat[:] = [0, 0, GRIPPER_Z_UP]
                        v.cam.elevation = -25
                if 'drop_obj' in cmds:
                    drop_object(data, cmds['drop_obj'],
                                tape_side=cmds.get('drop_tape_side', False),
                                rotation_deg=cmds.get('drop_rot'),
                                model=model, scale=cmds.get('drop_scale'),
                                hand_down=state['hand_down'],
                                translation_m=cmds.get('drop_trans'))
                    print(f"  >> [Panel] Dropped {cmds['drop_obj']}")
                if cmds.get('park'):
                    park_all_objects(data, model); print("  >> [Panel] Parked")
                if 'set_weight' in cmds:
                    wn,wg = cmds['set_weight']
                    set_object_weight(model, wn, wg)
                    print(f"  >> {wn} → {wg:.1f} g")
                if 'set_material' in cmds:
                    oname, fr, tc, dmax = cmds['set_material']
                    set_object_material(model, oname, friction=fr,
                                        solref_tc=tc, solimp_dmax=dmax)
                if 'reset_material' in cmds:
                    reset_object_material(model, cmds['reset_material'])
                    print(f"  >> Material reset: {cmds['reset_material']}")
                if 'set_splay' in cmds:
                    state['splay_deg'] = float(np.clip(cmds['set_splay'], 0, 30))
                if 'set_max_motor_deg' in cmds:
                    state['max_motor_deg'] = float(np.clip(cmds['set_max_motor_deg'], 450, 700))
                if 'set_finger_contact' in cmds:
                    sr, dm, wd, mg = cmds['set_finger_contact']
                    set_finger_solref(model, sr)
                    set_finger_solimp(model, dmax=dm, width=wd)
                    set_finger_margin(model, mg)
                if 'set_finger_solref_dr' in cmds:
                    set_finger_solref_damp(model, cmds['set_finger_solref_dr'])
                if 'set_finger_damping' in cmds:
                    set_finger_damping(model, cmds['set_finger_damping'])
                if 'set_finger_kp' in cmds:
                    set_actuator_kp(model, cmds['set_finger_kp'])
                if 'set_contact_scale' in cmds:
                    CONTACT_SCALE = float(np.clip(cmds['set_contact_scale'], 0.02, 0.60))
                if 'set_defl_gain' in cmds:
                    DEFLECTION_GAIN = float(np.clip(cmds['set_defl_gain'], 50, 800))
                if 'set_grasp_speed' in cmds:
                    state['grasp_speed'] = float(np.clip(cmds['set_grasp_speed'], 0.1, 2.0))
                if 'set_noslip_iter' in cmds:
                    model.opt.noslip_iterations = int(np.clip(cmds['set_noslip_iter'], 0, 10))
                if 'set_solver_iter' in cmds:
                    model.opt.iterations = int(np.clip(cmds['set_solver_iter'], 50, 500))

                if state['grasping']:
                    state['closure_pct'] += state['grasp_speed']
                    if state['closure_pct'] >= 100:
                        state['closure_pct']=100; state['grasping']=False
                        print(f"  >> Grasped ({state['max_motor_deg']:.0f}°)")
                        print_segment_forces(model, data, state)
                elif state['releasing']:
                    state['closure_pct'] -= state['grasp_speed']
                    if state['closure_pct'] <= 0:
                        state['closure_pct']=0; state['releasing']=False
                        print("  >> Released")
                elif state['auto_cycle']:
                    state['closure_pct'] += state['direction']*state['speed']
                    if state['closure_pct']>=100: state['closure_pct']=100; state['direction']=-1
                    elif state['closure_pct']<=0: state['closure_pct']=0; state['direction']=1

                apply_controls()
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                logger.log(model, data, state)

                now = data.time
                if now - last_print >= 3.0:
                    mmd = state['max_motor_deg']
                    md  = state['closure_pct']/100*mmd
                    sf  = get_segment_forces(model, data)
                    parts = []
                    for f in FINGERS:
                        s3 = sf[(f,3)]; s2 = sf[(f,2)]; s1 = sf[(f,1)]
                        if s1+s2+s3 > 0.1:
                            parts.append(f"{f}=[{s3:.1f},{s2:.1f},{s1:.1f}]")
                    fs = '  '.join(parts) if parts else 'no contact'
                    st = ' [GRASP]' if state['grasping'] else ' [REL]' if state['releasing'] else ''
                    hd = ' ↓DOWN' if state['hand_down'] else ''
                    sp = f' splay={state["splay_deg"]:.0f}°' if state['splay_deg'] > 0.1 else ''
                    print(f"  t={now:.1f}s  {md:.0f}°/{mmd:.0f}°  {state['closure_pct']:.1f}%{st}{hd}{sp}")
                    print(f"    force(g) [s3,s2,s1]: {fs}")
                    last_print = now

                if panel_obj and now - last_panel >= 0.2:
                    try:
                        panel_obj.refresh_readout(
                            state['closure_pct'], state['mode'], state['tension'],
                            state['selected'], state['grasping'], state['releasing'],
                            state['auto_cycle'], state['hand_down'],
                            state['splay_deg'], state['max_motor_deg'])
                    except: pass
                    last_panel = now

            v.sync()
            dt = model.opt.timestep*n_substeps - (time.time()-t0)
            if dt > 0: time.sleep(dt)

    print("\nDone.")
    print(f"  {len(logger.rows)} data rows logged.")
    logger.save_excel()


if __name__ == '__main__':
    main()