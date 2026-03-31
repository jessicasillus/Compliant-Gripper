"""
Gripper Simulation Python Script
Uses calibration data file to plot predicted shapes
User uploads existing data file

Jessica Sillus
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from mpl_toolkits.mplot3d import Axes3D   
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter1d
from grip_objects import build_object, OBJECT_NAMES

l    = 25e-3     #[m]
l0   = 25e-3     #[m]
lj   = 15e-3     #[m]
h    = 6.5e-3    #[m]
bw   = 10e-3     #[m]
lh   = 12e-3     #[m]
E    = 67e6      # youngs modulus TPU [Pa]

# joint thickness array
JH = np.array([3.25e-3, 3.25e-3, 3.50e-3, 3.50e-3]) # [m]
# stiffness
_I  = (1.0/12.0) * bw * JH**3
k1, k2, k3, k4 = E * _I / lj
E_EFFECTIVE    = 2.5e6    # effective joint modulus for solver  [Pa]
_k1, _k2, _k3, _k4 = E_EFFECTIVE * _I / lj   
EFFECTIVE_RP      = 0.015      # m (pulley radius)
C_TENDON = 0.005
PALM_HALF_SPAN    = 7.5e-3     # half of 15 mm AB-to-C gap [m]
FINGER_SEPARATION = 20e-3      # lateral A-B spacing        [m]
MAX_MOTOR_DEG     = 210.0      # full-closure motor travel  [deg]

PREDICTION_GAIN = 0.40
FORCE_EFFECT = 0.10
PRETENSION_DEG = 16.0

def _solve_pretension_angles(pretension_deg=None, gain=None,
                             _lk_override=None):
    """Solve for resting finger shape from tendon pretension.
    Returns (th1, th2, th3, th4) in radians.
    """
    if pretension_deg is None:
        pretension_deg = PRETENSION_DEG
    if gain is None:
        gain = PREDICTION_GAIN
    dphi = np.deg2rad(pretension_deg) * gain
    if dphi <= 0:
        return 0.0, 0.0, 0.0, 0.0
    th1, th2, th3, th4, _ = solve_angles(dphi, _lk_override=_lk_override)
    return th1, th2, th3, th4

FINGER_CONFIG = {
    'A': ( 90.0, +FINGER_SEPARATION / 2),
    'B': ( 90.0, -FINGER_SEPARATION / 2),
    'C': (270.0,  0.0),
}

def get_effective_finger_config(splay_deg=0.0):
    s = float(splay_deg)
    return {
        'A': ( 90.0 + s, +FINGER_SEPARATION / 2),
        'B': ( 90.0 - s, -FINGER_SEPARATION / 2),
        'C': (270.0,      0.0),
    }

finger_motor = {'A': 'm2', 'B': 'm2', 'C': 'm2'}

sensor_cols  = {
    'A': ('A1', 'A2', 'A3'),
    'B': ('B1', 'B2', 'B3'),
    'C': ('C1', 'C2', 'C3'),
}
position_col = {'m1': 'm1_position_deg', 'm2': 'm2_position_deg'}
current_col  = {'m1': 'm1_current',     'm2': 'm2_current'}
finger_direction          = {'A': +1,   'B': +1,   'C': +1}
finger_resting_percentile = {'A': 0.05, 'B': 0.05, 'C': 0.05}

LINK_FC  = np.array([0.88, 0.55, 0.10])
JOINT_FC = [
    np.array([0.55, 0.30, 0.05]),         
    np.array([0.62, 0.36, 0.08]),                
    np.array([0.70, 0.42, 0.10]),           
    np.array([0.78, 0.50, 0.14]),          
]
LINK_EC  = (0.35, 0.18, 0.02)                
BODY_FC  = (0.78, 0.78, 0.80)                  
BODY_EC  = (0.48, 0.48, 0.50) 

FINGER_CLR = {
    'A': '#d07010',   # warm orange
    'B': '#b85800',   # deeper amber
    'C': '#8e4400',   # brown-orange
}

FIG_BG_3D   = '#ebebeb'
AX_BG_3D    = '#f0f0f0'
PANE_FC_3D  = '#d8d8d8'
PANE_EC_3D  = '#c2c2c2'
GRID_CLR_3D = '#c8c8c8'
TICK_CLR    = '#444'
LABEL_CLR   = '#333'

ANIM_STEP           = 0.5
ANIM_MS             = 30
ANIM_SMOOTH_FRAMES  = 8
ANIM_INTERVAL_MS    = 50

SHOW_GRIP_OBJ = False
GRIP_OBJ_R    = 12e-3
GRIP_OBJ_H    = 60e-3

BASELINE_SECONDS    = 3.0   
STUCK_DROP          = 50      
DEADBAND            = 2.0     
SMOOTH_WINDOW       = 7       
FALLBACK_SCALE      = 5.0     

CAL_POLY_MODE = 'poly3'

PER_SENSOR_SCALE = {
    'A2': 10.0 / 5.4,    
    'A3': 10.0 / 1.6,    
    'B3': 20.0 / 8.5,    
    'C3': 20.0 / 1.0,    
}

ARROW_G_SCALE = 10e-4   # m per gram
FORCE_MAX_G   = 200.0  # grams at which link color is fully red

BODY_DROP = 0.012  # [m]

_POLY3_COEFFS = {}
_POLY2_COEFFS = {} 
_POLY_LOADED  = False

_CAL_XP     = {}
_CAL_FP     = {}
_CAL_LOADED = False


def load_calibration(path=None):

    global _POLY3_COEFFS, _POLY2_COEFFS, _POLY_LOADED
    global _CAL_XP, _CAL_FP, _CAL_LOADED

    if path is None:
        print(f'  No calibration file — using fallback ({FALLBACK_SCALE} g/ADC).')
        _CAL_LOADED = False
        _POLY_LOADED = False
        return

    try:
        xl = pd.ExcelFile(path)

        # ── 1. Polynomial fits (Best_Fit or All_Fits) ──────────────────
        fit_sheet = None
        if 'All_Fits' in xl.sheet_names:
            fit_sheet = 'All_Fits'
        elif 'Best_Fit' in xl.sheet_names:
            fit_sheet = 'Best_Fit'

        if fit_sheet is not None:
            df_fits = pd.read_excel(xl, sheet_name=fit_sheet)
            df_fits.columns = df_fits.columns.str.strip()
            fit_col = 'Fit_Type' if 'Fit_Type' in df_fits.columns else 'Best_Fit'
            _POLY3_COEFFS.clear()
            _POLY2_COEFFS.clear()
            for _, row in df_fits.iterrows():
                sensor   = str(row['Sensor']).strip()
                fit_type = str(row.get(fit_col, '')).strip()
                c3 = float(row.get('c3', 0) or 0)
                c2 = float(row.get('c2', 0) or 0)
                c1 = float(row.get('c1', 0) or 0)
                c0 = float(row.get('c0', 0) or 0)
                if fit_type == 'poly3':
                    _POLY3_COEFFS[sensor] = (c3, c2, c1, c0)
                elif fit_type == 'poly2':
                    _POLY2_COEFFS[sensor] = (c2, c1, c0)
            _POLY_LOADED = bool(_POLY3_COEFFS or _POLY2_COEFFS)
            if _POLY_LOADED:
                print(f'  Poly fits loaded ({fit_sheet}): '
                      f'poly3={sorted(_POLY3_COEFFS.keys())}  '
                      f'poly2={sorted(_POLY2_COEFFS.keys())}')

        it_sheet = None
        if 'Interpolation_Table' in xl.sheet_names:
            it_sheet = 'Interpolation_Table'
        elif not _POLY_LOADED:
            it_sheet = xl.sheet_names[0]

        if it_sheet is not None:
            df_it = pd.read_excel(xl, sheet_name=it_sheet)
            df_it.columns = df_it.columns.str.strip()
            col_map = {}
            for col in df_it.columns:
                lc = col.lower().replace(' ', '_').replace('-', '_')
                if lc == 'sensor':                              col_map['sensor'] = col
                if lc in ('weight_val', 'weight_g', 'grams'):  col_map['weight'] = col
                if lc in ('mean_delta', 'delta', 'adc_delta'): col_map['delta']  = col
            if all(k in col_map for k in ('sensor', 'weight', 'delta')):
                df_it = df_it[[col_map['sensor'],
                               col_map['weight'],
                               col_map['delta']]].copy()
                df_it.columns = ['sensor', 'weight', 'delta']
                df_it = df_it.dropna().sort_values(['sensor', 'delta'])
                _CAL_XP.clear(); _CAL_FP.clear()
                for sensor, grp in df_it.groupby('sensor'):
                    grp  = grp.sort_values('delta')
                    xp   = grp['delta'].values.astype(float)
                    fp   = grp['weight'].values.astype(float)
                    mask = np.concatenate([[True], np.diff(xp) > 0])
                    _CAL_XP[sensor] = xp[mask]
                    _CAL_FP[sensor] = fp[mask]
                _CAL_LOADED = True
                print(f'  Interpolation table loaded ({it_sheet}): '
                      f'{sorted(_CAL_XP.keys())}')

        if not _POLY_LOADED and not _CAL_LOADED:
            raise ValueError('No usable calibration data found in file.')

    except Exception as exc:
        print(f'  WARNING: calibration load failed ({exc})')
        print(f'  Falling back to linear scale ({FALLBACK_SCALE} g/ADC).')
        _CAL_LOADED  = False
        _POLY_LOADED = False


def adc_to_grams(sensor, delta):

    scalar = np.isscalar(delta)
    delta  = np.atleast_1d(np.asarray(delta, dtype=float)).copy()
    result = np.zeros_like(delta)

    pos = delta > 0
    if pos.any():
        d = delta[pos]

        prefer_p3 = (CAL_POLY_MODE != 'poly2')

        if prefer_p3 and _POLY_LOADED and sensor in _POLY3_COEFFS:
            c3, c2, c1, c0 = _POLY3_COEFFS[sensor]
            result[pos] = np.maximum(0.0, c3*d**3 + c2*d**2 + c1*d + c0)

        elif _POLY_LOADED and sensor in _POLY2_COEFFS:
            c2, c1, c0 = _POLY2_COEFFS[sensor]
            result[pos] = np.maximum(0.0, c2*d**2 + c1*d + c0)

        elif not prefer_p3 and _POLY_LOADED and sensor in _POLY3_COEFFS:
            c3, c2, c1, c0 = _POLY3_COEFFS[sensor]
            result[pos] = np.maximum(0.0, c3*d**3 + c2*d**2 + c1*d + c0)

        elif _CAL_LOADED and sensor in _CAL_XP:
            xp, fp = _CAL_XP[sensor], _CAL_FP[sensor]
            r = np.interp(d, xp, fp, left=0.0, right=fp[-1])
            if len(xp) >= 2:
                slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
                hi    = d > xp[-1]
                r[hi] = fp[-1] + (d[hi] - xp[-1]) * slope
            result[pos] = np.maximum(0.0, r)

        elif sensor in PER_SENSOR_SCALE:
            result[pos] = d * PER_SENSOR_SCALE[sensor]

        else:
            result[pos] = d * FALLBACK_SCALE

    return float(result[0]) if scalar else result


def calibrate_sensors(df):

    MIN_VALID_RAW = 100   # ignore stuck/zero readings when building baseline
    all_sensors   = [c for cols in sensor_cols.values() for c in cols]
    baselines     = {}
    df_cal        = df.copy()

    # Identify the unloaded window at the start of the recording
    t_start = df['timestamp'].iloc[0]
    early   = df[df['timestamp'] <= t_start + BASELINE_SECONDS]

    for s in all_sensors:
        col = df_cal[s].copy().astype(float)

        # Baseline = mean of early unloaded rows
        early_vals = early[s].values.astype(float)
        valid      = early_vals[early_vals >= MIN_VALID_RAW]
        baseline   = float(np.mean(valid)) if len(valid) > 0 \
                     else float(col.quantile(0.10))  # last-resort fallback
        baselines[s] = baseline

        # Repair any stuck/dead samples before computing delta
        col[col < baseline - STUCK_DROP] = np.nan
        col = col.interpolate(method='linear', limit_direction='both')

        # Optional smoothing to reduce noise
        if SMOOTH_WINDOW > 1:
            col = col.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()

        # delta = |baseline - raw| 
        delta = np.abs(baseline - col)
        delta[delta < DEADBAND] = 0.0

        df_cal[s] = adc_to_grams(s, delta.values).clip(min=0.0)

    return df_cal, baselines


def compute_resting_positions(df):
    resting = {}
    for finger in ('A', 'B', 'C'):
        motor = finger_motor[finger]
        col   = position_col[motor]
        if col in df.columns:
            resting[finger] = df[col].quantile(finger_resting_percentile[finger])
    return resting


def _kinematic_equations(x, DeltaPhi, F1, F2, F3, F4, lk1, lk2, lk3, lk4, c_t):
   
    theta1, theta2, theta3, theta4, Fc = x
    eq = np.zeros(5)

    eq[0] = (F4*(l/2)
             - Fc*h
             + lk4*theta4)

    eq[1] = (F3*(l/2)
             + F4*(l/2) + F4*np.cos(theta4)*l
             - 2*Fc*h
             + lk3*theta3 + lk4*theta4)

    eq[2] = (F2*(l/2)
             + F3*np.sin(theta3)*(l/2*np.sin(theta3))
             + F3*np.cos(theta3)*(l/2*np.cos(theta3) + l)
             + F4*np.sin(theta4)*(l/2*np.sin(theta4) + l*np.sin(theta3))
             + F4*np.cos(theta4)*(l/2*np.cos(theta4) + l*np.cos(theta3) + l)
             - 3*Fc*h
             + lk2*theta2 + lk3*theta3 + lk4*theta4)

    eq[3] = (F1*(l/2)
             + F2*np.sin(theta2)*(l/2*np.sin(theta2))
             + F2*np.cos(theta2)*(l/2*np.cos(theta2) + l)
             + F3*np.sin(theta3)*(l/2*np.sin(theta3) + l*np.sin(theta2))
             + F3*np.cos(theta3)*(l/2*np.cos(theta3) + l*np.cos(theta2) + l)
             + F4*np.sin(theta4)*(l/2*np.sin(theta4) + l*np.sin(theta3) + l*np.sin(theta2))
             + F4*np.cos(theta4)*(l/2*np.cos(theta4) + l*np.cos(theta3) + l*np.cos(theta2) + l)
             - 4*Fc*h
             + lk1*theta1 + lk2*theta2 + lk3*theta3 + lk4*theta4)

    eq[4] = h*(theta1 + theta2 + theta3 + theta4) - EFFECTIVE_RP*DeltaPhi + c_t*Fc
    return eq


def solve_angles(DeltaPhi, F1=0.0, F2=0.0, F3=0.0, F4=0.0, warm_start=None,
                 _lk_override=None, _ct_override=None):

    lk1, lk2, lk3, lk4 = _lk_override if _lk_override else (_k1, _k2, _k3, _k4)
    c_t = _ct_override if _ct_override is not None else C_TENDON
    if DeltaPhi <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if warm_start is not None:
        x0 = warm_start
    else:
        total_theta = EFFECTIVE_RP * DeltaPhi / h
        per_joint   = max(total_theta / 4.0, 1e-4)
        x0 = [per_joint, per_joint, per_joint, per_joint, (lk1 * per_joint) / h]
    try:
        x, _, ier, _ = fsolve(_kinematic_equations, x0,
                              args=(DeltaPhi, F1, F2, F3, F4, lk1, lk2, lk3, lk4, c_t),
                              full_output=True)
        if ier != 1:
            total_theta = EFFECTIVE_RP * DeltaPhi / h
            per_joint   = max(total_theta / 4.0, 1e-4)
            x0_fresh    = [per_joint, per_joint, per_joint, per_joint,
                           (lk1 * per_joint) / h]
            x, _, ier, _ = fsolve(_kinematic_equations, x0_fresh,
                                  args=(DeltaPhi, F1, F2, F3, F4, lk1, lk2, lk3, lk4, c_t),
                                  full_output=True)
            if ier != 1:
                return 0.0, 0.0, 0.0, 0.0, 0.0

        th1, th2, th3, th4, Fc = (float(x[0]), float(x[1]),
                                   float(x[2]), float(x[3]), float(x[4]))

        th1 = float(np.clip(th1, -np.pi/4, np.pi/2))
        th2 = float(np.clip(th2, -np.pi/4, np.pi/2))
        th3 = float(np.clip(th3, -np.pi/4, np.pi/2))
        th4 = float(np.clip(th4, -np.pi/4, np.pi/2))

        return th1, th2, th3, th4, Fc
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def forward_kinematics_2d(th1, th2, th3, th4):

    d = np.array
    def uv(a): return d([np.cos(a), np.sin(a)])
    p0  = d([0.0, 0.0])
    pB  = p0  + l0  * uv(0.0)
    pJ1 = pB  + lj  * uv(0.0)
    p1  = pJ1 + l   * uv(th1)
    pJ2 = p1  + lj  * uv(th1)
    p2  = pJ2 + l   * uv(th1 + th2)
    pJ3 = p2  + lj  * uv(th1 + th2)
    p3  = pJ3 + l   * uv(th1 + th2 + th3)
    pJ4 = p3  + lj  * uv(th1 + th2 + th3)
    p4  = pJ4 + l   * uv(th1 + th2 + th3 + th4)
    pm1 = pJ1 + (l/2) * uv(th1)
    pm2 = pJ2 + (l/2) * uv(th1 + th2)
    pm3 = pJ3 + (l/2) * uv(th1 + th2 + th3)
    pm4 = pJ4 + (l/2) * uv(th1 + th2 + th3 + th4)
    return p0, pB, pJ1, p1, pJ2, p2, pJ3, p3, pJ4, p4, pm1, pm2, pm3, pm4

def _box_faces(start, along, width_ax, height_ax, length, width, height):
    hw, hh = width / 2.0, height / 2.0
    end = start + length * along

    def _c(pt, sw, sh):
        return pt + sw * width_ax + sh * height_ax

    sc = [_c(start, sw, sh) for sw, sh in [(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)]]
    ec = [_c(end,   sw, sh) for sw, sh in [(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)]]
    return [
        sc,
        ec,
        [sc[0], sc[3], ec[3], ec[0]],
        [sc[1], sc[2], ec[2], ec[1]],
        [sc[0], sc[1], ec[1], ec[0]],
        [sc[3], sc[2], ec[2], ec[3]],
    ]


def _aabb_faces(cx, cy, cz, dx, dy, dz):
    v = np.array([
        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
        [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz],
    ])
    return [
        [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
        [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]],
    ]

def build_finger_mesh(th1, th2, th3, th4, mount_angle_deg, lat_off, link_forces_g=None, z_sign=1):
    phi      = np.deg2rad(mount_angle_deg)
    r_in     = np.array([-np.cos(phi), -np.sin(phi), 0.0])
    lat      = np.array([-np.sin(phi),  np.cos(phi), 0.0])
    base     = (np.array([PALM_HALF_SPAN * np.cos(phi),
                          PALM_HALF_SPAN * np.sin(phi), 0.0])
                + lat_off * lat)
    pos      = base.copy()
    along    = r_in.copy()
    up_local = np.array([0.0, 0.0, float(z_sign)])
    all_faces, all_colors = [], []

    _link_counter = [0]

    PALM_H = lh * 1.25

    def _add_link(length):
        nonlocal pos
        idx_l  = _link_counter[0]
        height = PALM_H if idx_l == 0 else lh
        faces  = _box_faces(pos, along, lat, up_local, length, bw, height)
        all_faces.extend(faces)
        if link_forces_g is not None and 2 <= idx_l <= 4:
            # link_forces_g = [C1, C2, C3]; L1=idx2→C1[0], L2=idx3→C2[1], L3/tip=idx4→C3[2]
            g_val = float(np.clip(link_forces_g[idx_l - 2], 0.0, FORCE_MAX_G))
            t  = g_val / FORCE_MAX_G
            fc = LINK_FC * (1.0 - t) + np.array([0.85, 0.05, 0.05]) * t
        else:
            fc = LINK_FC
        all_colors.extend([fc] * 6)
        _link_counter[0] += 1
        pos = pos + length * along

    def _add_joint(joint_idx):
        nonlocal pos
        faces = _box_faces(pos, along, lat, up_local, lj/2, bw, JH[joint_idx])
        all_faces.extend(faces)
        all_colors.extend([JOINT_FC[joint_idx]] * 6)
        pos = pos + (lj/2) * along

    def _add_joint_end(joint_idx):
        nonlocal pos
        faces = _box_faces(pos, along, lat, up_local, lj/2, bw, JH[joint_idx])
        all_faces.extend(faces)
        all_colors.extend([JOINT_FC[joint_idx]] * 6)
        pos = pos + (lj/2) * along

    def _bend(theta):
        nonlocal along, up_local
        c, s = np.cos(theta), np.sin(theta)
        along, up_local = c*along + s*up_local, -s*along + c*up_local

    _add_link(l0)

    for i, theta in enumerate((th1, th2, th3, th4)):
        _add_joint(i)
        _bend(theta)
        _add_joint_end(i)
        _add_link(l)

    return all_faces, all_colors


def _fingertip_3d(th1, th2, th3, th4, mount_angle_deg, lat_off, z_sign=1):
    phi      = np.deg2rad(mount_angle_deg)
    r_in     = np.array([-np.cos(phi), -np.sin(phi), 0.0])
    lat      = np.array([-np.sin(phi),  np.cos(phi), 0.0])
    base     = (np.array([PALM_HALF_SPAN * np.cos(phi),
                          PALM_HALF_SPAN * np.sin(phi), 0.0])
                + lat_off * lat)
    along    = r_in.copy()
    up_local = np.array([0.0, 0.0, float(z_sign)])
    pos      = base + l0 * along
    for theta in (th1, th2, th3, th4):  # palm→tip order
        pos += (lj/2) * along
        c, s = np.cos(theta), np.sin(theta)
        along, up_local = c*along + s*up_local, -s*along + c*up_local
        pos += (lj/2) * along
        pos += l * along
    return pos


def build_3d_figure(z_sign=1, object_name='None'):

    fig = plt.figure(figsize=(14, 10), facecolor=FIG_BG_3D)
    title_txt = fig.suptitle('3-Finger Compliant Gripper',
                 fontsize=13, fontweight='bold', fontfamily='monospace',
                 color=LABEL_CLR, y=0.98)

    # 3-D axes: left 74% width, leaving room for side panel
    ax = fig.add_axes([0.01, 0.13, 0.74, 0.84], projection='3d')
    ax.set_facecolor(AX_BG_3D)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(PANE_FC_3D)
        pane.set_edgecolor(PANE_EC_3D)
    ax.tick_params(colors=TICK_CLR, labelsize=7)
    ax.set_xlabel('X  [m]', labelpad=4, color=TICK_CLR, fontsize=8)
    ax.set_ylabel('Y  [m]', labelpad=4, color=TICK_CLR, fontsize=8)
    ax.set_zlabel('Z  [m]', labelpad=4, color=TICK_CLR, fontsize=8)
    ax.xaxis._axinfo['grid']['color'] = GRID_CLR_3D
    ax.yaxis._axinfo['grid']['color'] = GRID_CLR_3D
    ax.zaxis._axinfo['grid']['color'] = GRID_CLR_3D
    ax.view_init(elev=22, azim=200)

    reach = PALM_HALF_SPAN + (l + lj) * 4 + l0 + 0.015
    R = max(reach, FINGER_SEPARATION / 2 + 0.025)
    H = (l + lj) * 4 + l0 + 0.018
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    if z_sign > 0:
        ax.set_zlim(-0.070, H)
    else:
        ax.set_zlim(-H, 0.070)

    bdx = FINGER_SEPARATION / 2 + bw / 2 + 0.006
    bdy = PALM_HALF_SPAN + 0.012
    bdz = 0.022
    body_z1 = z_sign * (-bdz - BODY_DROP)
    body_z2 = z_sign * (-2 * bdz - 0.005 - BODY_DROP)
    body1 = Poly3DCollection(
        _aabb_faces(0, 0, body_z1, bdx, bdy, bdz),
        alpha=1.0, facecolor=BODY_FC, edgecolor=BODY_EC,
        linewidth=0.8, zsort='min')
    body2 = Poly3DCollection(
        _aabb_faces(0, 0, body_z2, bdx + 0.007, bdy + 0.007, 0.005),
        alpha=1.0, facecolor=(0.62, 0.62, 0.64),
        edgecolor=(0.40, 0.40, 0.42),
        linewidth=0.6, zsort='min')
    ax.add_collection3d(body1)
    ax.add_collection3d(body2)
    body_cols = [body1, body2]

    obj_col = None
    obj_faces, obj_fcolors = build_object(object_name, z_sign=z_sign)
    if obj_faces is not None:
        obj_col = Poly3DCollection(
            obj_faces, facecolors=obj_fcolors,
            edgecolors=[(0.25, 0.25, 0.25, 0.5)] * len(obj_faces),
            linewidth=0.3, alpha=0.92, zsort='average')
        ax.add_collection3d(obj_col)

    _pt = _solve_pretension_angles()
    finger_cols = {}
    for f, (ang_deg, lat_off) in FINGER_CONFIG.items():
        faces, fcolors = build_finger_mesh(
            _pt[0], _pt[1], _pt[2], _pt[3],
            ang_deg, lat_off, z_sign=z_sign)
        col = Poly3DCollection(faces, facecolors=fcolors,
                               edgecolors=[LINK_EC] * len(faces),
                               linewidth=0.45, zsort='min', alpha=1.0)
        ax.add_collection3d(col)
        finger_cols[f] = col

    tip_dots = {}
    for f in FINGER_CONFIG:
        dot, = ax.plot([], [], [], 'o', color='white', markersize=7,
                       markeredgecolor=FINGER_CLR[f],
                       markeredgewidth=1.6, zorder=9, linestyle='None')
        tip_dots[f] = dot

    ax_side = fig.add_axes([0.77, 0.42, 0.22, 0.55])
    ax_side.axis('off')
    info_txt = ax_side.text(
        0.05, 0.98, '', transform=ax_side.transAxes,
        fontsize=9.5, fontfamily='monospace', color='#222',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                  edgecolor='#bbb', linewidth=0.8))

    ax_orient = fig.add_axes([0.78, 0.330, 0.10, 0.075])
    ax_orient.set_title('Orientation', fontsize=8, fontfamily='monospace',
                        color=LABEL_CLR, pad=3)

    n_obj = len(OBJECT_NAMES)
    ax_objsel = fig.add_axes([0.78, 0.158, 0.12, 0.04 + n_obj * 0.025])
    ax_objsel.set_title('Grip Object', fontsize=8, fontfamily='monospace',
                        color=LABEL_CLR, pad=3)


    scene_state = {
        'z_sign':       z_sign,
        'object_name':  object_name,
        'ax':           ax,
        'body_cols':    body_cols,
        'obj_col':      obj_col,
        'ax_orient':    ax_orient,
        'ax_objsel':    ax_objsel,
        'obj_offset':   [0.0, 0.0, 0.0],
        'obj_scale':    1.0,
        'obj_rotation': [0.0, 0.0, 0.0],   # [Rx, Ry, Rz] degrees
    }


    _ctrl_x  = 0.78
    _ctrl_w  = 0.19
    _ctrl_sh = 0.015   # slider height
    _ctrl_dy = 0.022   # row pitch

    def _cy(row):          # row 0 = bottom-most
        return 0.005 + row * _ctrl_dy

    ax_rz = fig.add_axes([_ctrl_x, _cy(0), _ctrl_w, _ctrl_sh])
    ax_ry = fig.add_axes([_ctrl_x, _cy(1), _ctrl_w, _ctrl_sh])
    ax_rx = fig.add_axes([_ctrl_x, _cy(2), _ctrl_w, _ctrl_sh])
    ax_os = fig.add_axes([_ctrl_x, _cy(3), _ctrl_w, _ctrl_sh])
    ax_oz = fig.add_axes([_ctrl_x, _cy(4), _ctrl_w, _ctrl_sh])
    ax_oy = fig.add_axes([_ctrl_x, _cy(5), _ctrl_w, _ctrl_sh])
    ax_ox = fig.add_axes([_ctrl_x, _cy(6), _ctrl_w, _ctrl_sh])

    from matplotlib.widgets import Slider as _Sl
    sl_ox = _Sl(ax_ox, 'Obj X',  -0.10, 0.10,  valinit=0.0,  color='#609060')
    sl_oy = _Sl(ax_oy, 'Obj Y',  -0.10, 0.10,  valinit=0.0,  color='#609060')
    sl_oz = _Sl(ax_oz, 'Obj Z',  -0.10, 0.10,  valinit=0.0,  color='#609060')
    sl_os = _Sl(ax_os, 'Scale',   0.2,  3.0,   valinit=1.0,  color='#906060')
    sl_rx = _Sl(ax_rx, 'Rot X', -180.0, 180.0, valinit=0.0,  color='#506090')
    sl_ry = _Sl(ax_ry, 'Rot Y', -180.0, 180.0, valinit=0.0,  color='#506090')
    sl_rz = _Sl(ax_rz, 'Rot Z', -180.0, 180.0, valinit=0.0,  color='#506090')
    for _s in (sl_ox, sl_oy, sl_oz, sl_os, sl_rx, sl_ry, sl_rz):
        _s.label.set_fontfamily('monospace'); _s.label.set_fontsize(7)
        _s.valtext.set_fontfamily('monospace'); _s.valtext.set_fontsize(7)

    scene_state['sl_ox'] = sl_ox
    scene_state['sl_oy'] = sl_oy
    scene_state['sl_oz'] = sl_oz
    scene_state['sl_os'] = sl_os
    scene_state['sl_rx'] = sl_rx
    scene_state['sl_ry'] = sl_ry
    scene_state['sl_rz'] = sl_rz

    overlay_texts = []
    return (fig, ax, ax_side, finger_cols, tip_dots, info_txt, title_txt,
            overlay_texts, scene_state)


def update_3d(fig3d, finger_cols, tip_dots, angles_per_finger, z_sign=1,
              splay_deg=0.0):

    cfg = get_effective_finger_config(splay_deg)
    for f, (ang_deg, lat_off) in cfg.items():
        th1, th2, th3, th4, Fc, forces_g = angles_per_finger[f]
        faces, fcolors = build_finger_mesh(th1, th2, th3, th4, ang_deg, lat_off,
                                           link_forces_g=forces_g, z_sign=z_sign)
        col = finger_cols[f]
        col.set_verts(faces)
        col.set_facecolor(fcolors)
        col.set_edgecolor([LINK_EC] * len(faces))
        tip = _fingertip_3d(th1, th2, th3, th4, ang_deg, lat_off, z_sign=z_sign)
        tip_dots[f].set_data_3d([tip[0]], [tip[1]], [tip[2]])
    fig3d.canvas.draw_idle()


def launch_simulation():
    from matplotlib.widgets import (Button as _Button, Slider as _Slider,
                                    TextBox as _TB)

    scene = {'z_sign': 1, 'object_name': 'None'}

    (fig3d, ax3d, ax_side,
     finger_cols, tip_dots, info_txt, title_txt, _,
     scene_state) = build_3d_figure(z_sign=scene['z_sign'],
                                     object_name=scene['object_name'])
    title_txt.set_text('3-Finger Compliant Gripper — Simulation')

    orient_radio = mwidgets.RadioButtons(
        scene_state['ax_orient'], ['Hand Up', 'Hand Down'],
        active=0, activecolor='#2060b8')
    for lbl in orient_radio.labels:
        lbl.set_fontsize(8); lbl.set_fontfamily('monospace')

    obj_radio = mwidgets.RadioButtons(
        scene_state['ax_objsel'], OBJECT_NAMES,
        active=0, activecolor='#c86010')
    for lbl in obj_radio.labels:
        lbl.set_fontsize(7.5); lbl.set_fontfamily('monospace')

    fig2d = plt.figure(figsize=(18, 10), facecolor='white')
    fig2d.suptitle('Simulation — Finger Shape Viewer',
                   fontsize=14, fontweight='bold', fontfamily='monospace')

    gs = fig2d.add_gridspec(3, 1,
                            left=0.33, right=0.99,
                            top=0.93, bottom=0.07,
                            hspace=0.38)
    ax_A = fig2d.add_subplot(gs[0])
    ax_B = fig2d.add_subplot(gs[1])
    ax_C = fig2d.add_subplot(gs[2])

    ax_info = fig2d.add_axes([0.01, 0.52, 0.29, 0.41])
    ax_stat = fig2d.add_axes([0.01, 0.46, 0.29, 0.05])
    ax_info.axis('off'); ax_stat.axis('off')

    finger_axes = {'A': ax_A, 'B': ax_B, 'C': ax_C}

    _span = l0 + 4*(lj + l) + 0.01
    for f, ax in finger_axes.items():
        ax.set_xlim(-0.005, _span)
        ax.set_ylim(-0.075, 0.075)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title(f'Finger {f}', fontsize=10, fontfamily='monospace',
                     color=FINGER_CLR[f], fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(labelsize=7)

    link_colors  = ['#2060b8', '#3080d8', '#60a8f0', '#90c8ff']
    joint_colors = ['#8B5E3C', '#9B6E4C', '#AB7E5C', '#BB8E6C']


    artists = {}
    for f, ax in finger_axes.items():
        base_ln, = ax.plot([], [], '-', linewidth=10, color='#333',
                           solid_capstyle='round', zorder=2)
        j1, = ax.plot([], [], linewidth=5, color=joint_colors[0],
                      solid_capstyle='butt', zorder=2)
        j2, = ax.plot([], [], linewidth=5, color=joint_colors[1],
                      solid_capstyle='butt', zorder=2)
        j3, = ax.plot([], [], linewidth=5, color=joint_colors[2],
                      solid_capstyle='butt', zorder=2)
        j4, = ax.plot([], [], linewidth=5, color=joint_colors[3],
                      solid_capstyle='butt', zorder=2)
        l1, = ax.plot([], [], linewidth=8, color=link_colors[0],
                      solid_capstyle='round', zorder=3)
        l2, = ax.plot([], [], linewidth=8, color=link_colors[1],
                      solid_capstyle='round', zorder=3)
        l3, = ax.plot([], [], linewidth=8, color=link_colors[2],
                      solid_capstyle='round', zorder=3)
        l4, = ax.plot([], [], linewidth=8, color=link_colors[3],
                      solid_capstyle='round', zorder=3)
        jdots, = ax.plot([], [], 'o', markersize=6,
                         markerfacecolor='#444', markeredgecolor='#222',
                         markeredgewidth=1.0, zorder=4)
        tip,   = ax.plot([], [], 'o', markersize=11,
                         markerfacecolor='#cc2020', markeredgecolor='#880000',
                         markeredgewidth=1.4, zorder=5)
        artists[f] = dict(base=base_ln, j1=j1, j2=j2, j3=j3, j4=j4,
                          l1=l1, l2=l2, l3=l3, l4=l4,
                          joints=jdots, tip=tip,
                          arrows=[None, None, None])

    info_text = ax_info.text(
        0.03, 0.99, '', transform=ax_info.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                  edgecolor='#999', linewidth=1.2))
    status_text = ax_stat.text(
        0.03, 0.75, 'Simulation mode — type forces and adjust closure.',
        transform=ax_stat.transAxes,
        fontsize=8.5, fontfamily='monospace', color='darkorange', va='top')

    _sim_forces = {f: [0.0, 0.0, 0.0] for f in ('A', 'B', 'C')}
    _sim_tb_refs = []  

    ax_fhdr = fig2d.add_axes([0.02, 0.435, 0.28, 0.018])
    ax_fhdr.axis('off')
    ax_fhdr.text(0.0, 0.5, 'Contact Forces [g]', fontsize=8.5,
                 fontfamily='monospace', fontweight='bold', color=LABEL_CLR,
                 va='center', transform=ax_fhdr.transAxes)

    _ov_x0  = 0.08;   _ov_dx  = 0.075
    _ov_tbw = 0.060;   _ov_tbh = 0.022
    _ov_y0  = 0.410;   _ov_dy  = 0.028

    ax_chdr = fig2d.add_axes([0.01, _ov_y0 + 0.003, 0.28, 0.018])
    ax_chdr.axis('off')
    for ci, sname in enumerate(('S1 (prox)', 'S2 (mid)', 'S3 (dist)')):
        ax_chdr.text((_ov_x0 - 0.01 + ci * _ov_dx) / 0.28, 0.5,
                     sname, fontsize=6.5, fontfamily='monospace',
                     color='#555', ha='left', va='center',
                     transform=ax_chdr.transAxes)

    for ri, flabel in enumerate(('A', 'B', 'C')):
        y_row = _ov_y0 - ri * _ov_dy
        ax_rl = fig2d.add_axes([0.02, y_row, 0.04, _ov_tbh])
        ax_rl.axis('off')
        ax_rl.text(0.9, 0.5, f'{flabel}:', fontsize=8, fontfamily='monospace',
                   fontweight='bold', color=FINGER_CLR[flabel],
                   ha='right', va='center', transform=ax_rl.transAxes)
        for ci in range(3):
            ax_tb = fig2d.add_axes([_ov_x0 + ci * _ov_dx, y_row,
                                    _ov_tbw, _ov_tbh])
            tb = _TB(ax_tb, '', initial='0')
            tb.text_disp.set_fontfamily('monospace')
            tb.text_disp.set_fontsize(8)
            _sim_tb_refs.append(tb)

            def _make_fcb(finger, idx):
                def _on_submit(text):
                    try:
                        _sim_forces[finger][idx] = float(text)
                    except ValueError:
                        _sim_forces[finger][idx] = 0.0
                    _full_update(sl_closure.val)
                return _on_submit
            tb.on_submit(_make_fcb(flabel, ci))

    _LX  = 0.02;  _LW  = 0.28;  _SH  = 0.024;  _DY  = 0.040;  _FS  = 9

    def _sl(ax_pos, label, lo, hi, val, color):
        ax = fig2d.add_axes([_LX, ax_pos, _LW, _SH])
        sl = _Slider(ax, label, lo, hi, valinit=val, color=color)
        for _w in (sl.label, sl.valtext):
            _w.set_fontfamily('monospace'); _w.set_fontsize(_FS)
        return sl

    _y0 = 0.330   # top slider bottom edge
    sl_closure = _sl(_y0 - 0*_DY, 'Closure %',       0.0,  100.0, 0.0,              '#c86010')
    sl_gain    = _sl(_y0 - 1*_DY, 'Gain',             0.1,    3.0, PREDICTION_GAIN, '#c06010')
    sl_eeff    = _sl(_y0 - 2*_DY, 'E_eff [MPa]',      0.5,   30.0, E_EFFECTIVE/1e6, '#601060')
    sl_feff    = _sl(_y0 - 3*_DY, 'Force Effect',     0.0,    1.0, FORCE_EFFECT,    '#10a040')
    sl_pret    = _sl(_y0 - 4*_DY, 'Pretension [deg]', 0.0,   60.0, PRETENSION_DEG,  '#8050a0')
    sl_ct      = _sl(_y0 - 5*_DY, 'Tendon Compl.',    0.0,   0.05, C_TENDON,        '#a06050')
    sl_splay   = _sl(_y0 - 6*_DY, 'Splay A/B [°]',    0.0,   60.0, 0.0,             '#208080')

    tune = {'gain': PREDICTION_GAIN, 'e_eff': E_EFFECTIVE,
            'force_effect': FORCE_EFFECT, 'pretension': PRETENSION_DEG,
            'c_tendon': C_TENDON, 'splay_deg': 0.0}

    _BH = 0.030;  _BW = 0.088
    _by = _y0 - 7*_DY
    ax_play2   = fig2d.add_axes([_LX,                  _by, _BW, _BH])
    ax_reset2  = fig2d.add_axes([_LX + _BW + 0.005,    _by, _BW, _BH])
    ax_savefig = fig2d.add_axes([_LX + 2*(_BW + 0.005)+0.01, _by, _BW, _BH])

    btn_play2   = _Button(ax_play2,   '>>  Play',      color='#ddeeff', hovercolor='#bbddff')
    btn_reset2  = _Button(ax_reset2,  '<>  Reset',     color='#ddeeff', hovercolor='#bbddff')
    btn_savefig = _Button(ax_savefig, '[~] Save Fig',  color='#fff3cc', hovercolor='#ffe88a')
    for _b in (btn_play2, btn_reset2, btn_savefig):
        _b.label.set_fontfamily('monospace'); _b.label.set_fontsize(8)

    state = {'playing': False, 'timer': None, 'direction': 1}

    ax_sl    = fig3d.add_axes([0.12, 0.07, 0.60, 0.022])
    ax_play  = fig3d.add_axes([0.12, 0.025, 0.09, 0.035])
    ax_reset = fig3d.add_axes([0.23, 0.025, 0.09, 0.035])

    slider = mwidgets.Slider(ax_sl, 'Closure  %', 0.0, 100.0, valinit=0.0,
                             color='#c86010', track_color='#cccccc',
                             handle_style={'facecolor': '#e07020',
                                           'edgecolor': '#a04000',
                                           'size': 12})
    slider.label.set_color(LABEL_CLR);   slider.label.set_fontsize(9)
    slider.label.set_fontfamily('monospace')
    slider.valtext.set_color(TICK_CLR);  slider.valtext.set_fontsize(9)
    slider.valtext.set_fontfamily('monospace')

    btn_play  = mwidgets.Button(ax_play,  '>>  Play',  color='#ddd', hovercolor='#c4c4c4')
    btn_reset = mwidgets.Button(ax_reset, '<>  Reset', color='#ddd', hovercolor='#c4c4c4')
    for btn in (btn_play, btn_reset):
        btn.label.set_color(LABEL_CLR)
        btn.label.set_fontsize(9)
        btn.label.set_fontfamily('monospace')

    def _rebuild_object():
        zs = scene['z_sign']
        ax = scene_state['ax']
        if scene_state['obj_col'] is not None:
            try: scene_state['obj_col'].remove()
            except ValueError: pass
            scene_state['obj_col'] = None
        off = tuple(scene_state['obj_offset'])
        scl = scene_state['obj_scale']
        rot = tuple(scene_state['obj_rotation'])
        obj_faces, obj_fc = build_object(scene['object_name'], z_sign=zs,
                                         offset=off, scale=scl,
                                         rotation_deg=rot)
        if obj_faces is not None:
            col = Poly3DCollection(
                obj_faces, facecolors=obj_fc,
                edgecolors=[(0.25, 0.25, 0.25, 0.5)] * len(obj_faces),
                linewidth=0.3, alpha=0.92, zsort='average')
            ax.add_collection3d(col)
            scene_state['obj_col'] = col

    def _rebuild_body():
        zs = scene['z_sign']
        bdx = FINGER_SEPARATION / 2 + bw / 2 + 0.006
        bdy = PALM_HALF_SPAN + 0.012
        bdz = 0.022
        body_z1 = zs * (-bdz - BODY_DROP)
        body_z2 = zs * (-2 * bdz - 0.005 - BODY_DROP)
        scene_state['body_cols'][0].set_verts(
            _aabb_faces(0, 0, body_z1, bdx, bdy, bdz))
        scene_state['body_cols'][1].set_verts(
            _aabb_faces(0, 0, body_z2, bdx + 0.007, bdy + 0.007, 0.005))
        H = (l + lj) * 4 + l0 + 0.018
        if zs > 0: ax3d.set_zlim(-0.070, H)
        else:      ax3d.set_zlim(-H, 0.070)

    def _full_update(pct):
        """Solve kinematics and update BOTH 3-D and 2-D figures."""
        pct  = float(np.clip(pct, 0.0, 100.0))
        dphi = np.deg2rad(pct / 100.0 * MAX_MOTOR_DEG + tune['pretension']) * tune['gain']
        g_acc = 9.81
        _live_I = (1.0/12.0) * bw * JH**3
        _lk = tuple(tune['e_eff'] * _live_I / lj)

        angles_per_finger = {}
        for f in FINGER_CONFIG:
            fg = _sim_forces[f]
            F2 = fg[0] / 1000.0 * g_acc * tune['force_effect']
            F3 = fg[1] / 1000.0 * g_acc * tune['force_effect']
            F4 = fg[2] / 1000.0 * g_acc * tune['force_effect']
            th1, th2, th3, th4, Fc = solve_angles(dphi, 0.0, F2, F3, F4,
                                                   _lk_override=_lk,
                                                   _ct_override=tune['c_tendon'])
            angles_per_finger[f] = (th1, th2, th3, th4, Fc,
                                    [fg[0], fg[1], fg[2]])

        update_3d(fig3d, finger_cols, tip_dots, angles_per_finger,
                  z_sign=scene['z_sign'], splay_deg=tune['splay_deg'])

        lines3 = [
            f'Closure  {pct:5.1f} %',
            f'dPhi = {np.rad2deg(dphi):6.2f} deg  '
            f'(pret {tune["pretension"]:.1f}°)',
            '',
            '    J4     J3     J2     J1    Σ      Fc',
            '  ────────────────────────────────────────',
        ]
        for f in ('A', 'B', 'C'):
            th1, th2, th3, th4, Fc, _ = angles_per_finger[f]
            s = th1 + th2 + th3 + th4
            lines3.append(
                f'{f}  {np.rad2deg(th1):5.1f}  {np.rad2deg(th2):5.1f}'
                f'  {np.rad2deg(th3):5.1f}  {np.rad2deg(th4):5.1f}'
                f'  {np.rad2deg(s):5.1f}°  {Fc:.4f} N')
        lines3 += ['', 'Forces [g]  (S1  S2  S3)',
                    '  ─────────────────────────']
        for f in ('A', 'B', 'C'):
            fg = _sim_forces[f]
            lines3.append(f'{f}  [{fg[0]:5.1f}  {fg[1]:5.1f}  {fg[2]:5.1f}]')
        info_txt.set_text('\n'.join(lines3))

        for f, ax in finger_axes.items():
            th1, th2, th3, th4, Fc, g_vals = angles_per_finger[f]
            art = artists[f]

            (p0, pB, pJ1, p1,
             pJ2, p2, pJ3, p3,
             pJ4, p4,
             pm1, pm2, pm3, pm4) = forward_kinematics_2d(th1, th2, th3, th4)

            art['base'].set_data([p0[0],  pB[0]],  [p0[1],  pB[1]])
            art['j1'].set_data(  [pB[0],  pJ1[0]], [pB[1],  pJ1[1]])
            art['j2'].set_data(  [p1[0],  pJ2[0]], [p1[1],  pJ2[1]])
            art['j3'].set_data(  [p2[0],  pJ3[0]], [p2[1],  pJ3[1]])
            art['j4'].set_data(  [p3[0],  pJ4[0]], [p3[1],  pJ4[1]])
            art['l1'].set_data(  [pJ1[0], p1[0]],  [pJ1[1], p1[1]])
            art['l2'].set_data(  [pJ2[0], p2[0]],  [pJ2[1], p2[1]])
            art['l3'].set_data(  [pJ3[0], p3[0]],  [pJ3[1], p3[1]])
            art['l4'].set_data(  [pJ4[0], p4[0]],  [pJ4[1], p4[1]])
            art['joints'].set_data(
                [pB[0], pJ1[0], p1[0], pJ2[0], p2[0], pJ3[0], p3[0], pJ4[0]],
                [pB[1], pJ1[1], p1[1], pJ2[1], p2[1], pJ3[1], p3[1], pJ4[1]])
            art['tip'].set_data([p4[0]], [p4[1]])

            # Force arrows
            n2 = np.array([-np.sin(th1+th2),         np.cos(th1+th2)])
            n3 = np.array([-np.sin(th1+th2+th3),     np.cos(th1+th2+th3)])
            n4 = np.array([-np.sin(th1+th2+th3+th4), np.cos(th1+th2+th3+th4)])
            for q in art['arrows']:
                if q is not None: q.remove()
            for i, (pt, nv, gv) in enumerate(
                    zip([pm2, pm3, pm4], [n2, n3, n4], g_vals)):
                if gv > 0.5:
                    alen = gv * ARROW_G_SCALE
                    art['arrows'][i] = ax.quiver(
                        pt[0], pt[1], -alen*nv[0], -alen*nv[1],
                        color='#cc1111', linewidth=2,
                        scale=1, scale_units='xy', angles='xy', zorder=6)
                else:
                    art['arrows'][i] = None


        lines2 = [
            f'Closure: {pct:.1f} %',
            f'dPhi: {np.rad2deg(dphi):.2f} deg',
            f'Gain: {tune["gain"]:.2f}  E_eff: {tune["e_eff"]/1e6:.1f}MPa',
            f'Force Effect: {tune["force_effect"]:.2f}  '
            f'Pretension: {tune["pretension"]:.1f} deg',
            '',
            '    J1    J2    J3    J4   Σ',
            '  ──────────────────────────',
        ]
        for fi in ('A', 'B', 'C'):
            t1, t2, t3, t4, Fc, _ = angles_per_finger[fi]
            s = t1 + t2 + t3 + t4
            lines2.append(
                f'{fi} {np.rad2deg(t1):5.1f} {np.rad2deg(t2):5.1f}'
                f' {np.rad2deg(t3):5.1f} {np.rad2deg(t4):5.1f}'
                f' {np.rad2deg(s):5.1f}°  Fc={Fc:.4f}N')
        lines2 += ['', 'Forces [g]  (S1  S2  S3)',
                    '  ─────────────────────']
        for fi in ('A', 'B', 'C'):
            fg = _sim_forces[fi]
            lines2.append(f'{fi} [{fg[0]:5.1f} {fg[1]:5.1f} {fg[2]:5.1f}]')
        info_text.set_text('\n'.join(lines2))

        fig2d.canvas.draw_idle()

    def _sync_and_update(pct):
        """Sync both sliders and run full update."""
        slider.eventson = False
        slider.set_val(pct)
        slider.eventson = True
        sl_closure.eventson = False
        sl_closure.set_val(pct)
        sl_closure.eventson = True
        _full_update(pct)

    def on_slider_3d(val):   _sync_and_update(val)
    def on_slider_2d(val):   _sync_and_update(val)

    def on_gain(val):
        tune['gain'] = float(val)
        _full_update(sl_closure.val)
    def on_eeff(val):
        tune['e_eff'] = float(val) * 1e6
        _full_update(sl_closure.val)
    def on_feff(val):
        tune['force_effect'] = float(val)
        _full_update(sl_closure.val)
    def on_pret(val):
        tune['pretension'] = float(val)
        _full_update(sl_closure.val)
    def on_ct(val):
        tune['c_tendon'] = float(val)
        _full_update(sl_closure.val)
    def on_splay(val):
        tune['splay_deg'] = float(val)
        _full_update(sl_closure.val)

    def _step():
        val = sl_closure.val + state['direction'] * ANIM_STEP
        if val >= 100.0:
            val = 100.0; state['direction'] = -1
        elif val <= 0.0:
            val = 0.0;   state['direction'] = 1
        _sync_and_update(val)

    def on_play(event):
        if not state['playing']:
            state['playing'] = True
            btn_play.label.set_text('||  Pause')
            btn_play2.label.set_text('||  Pause')
            state['timer'] = fig3d.canvas.new_timer(interval=ANIM_MS)
            state['timer'].add_callback(_step)
            state['timer'].start()
        else:
            state['playing'] = False
            btn_play.label.set_text('>>  Play')
            btn_play2.label.set_text('>>  Play')
            if state['timer']:
                state['timer'].stop(); state['timer'] = None
        fig3d.canvas.draw_idle()
        fig2d.canvas.draw_idle()

    def on_reset(event):
        if state['playing']: on_play(None)
        state['direction'] = 1
        _sync_and_update(0.0)

    def on_orient(label):
        scene['z_sign'] = 1 if label == 'Hand Up' else -1
        _rebuild_body()
        _rebuild_object()
        _full_update(sl_closure.val)

    def on_objsel(label):
        scene['object_name'] = label
        _rebuild_object()
        fig3d.canvas.draw_idle()

    def on_obj_pos(_):
        scene_state['obj_offset'][0] = scene_state['sl_ox'].val
        scene_state['obj_offset'][1] = scene_state['sl_oy'].val
        scene_state['obj_offset'][2] = scene_state['sl_oz'].val
        scene_state['obj_scale']        = scene_state['sl_os'].val
        scene_state['obj_rotation'][0]  = scene_state['sl_rx'].val
        scene_state['obj_rotation'][1]  = scene_state['sl_ry'].val
        scene_state['obj_rotation'][2]  = scene_state['sl_rz'].val
        _rebuild_object()
        fig3d.canvas.draw_idle()

    def on_savefig(event):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(SAVE_DIR, exist_ok=True)
        for ext in ('png', 'pdf'):
            p = os.path.join(SAVE_DIR, f'gripper_sim_2d_{ts}.{ext}')
            fig2d.savefig(p, dpi=300, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
        print(f'  Saved → {SAVE_DIR}  (PNG + PDF)')

    slider.on_changed(on_slider_3d)
    sl_closure.on_changed(on_slider_2d)
    sl_gain.on_changed(on_gain)
    sl_eeff.on_changed(on_eeff)
    sl_feff.on_changed(on_feff)
    sl_pret.on_changed(on_pret)
    sl_ct.on_changed(on_ct)
    sl_splay.on_changed(on_splay)
    btn_play.on_clicked(on_play)
    btn_play2.on_clicked(on_play)
    btn_reset.on_clicked(on_reset)
    btn_reset2.on_clicked(on_reset)
    btn_savefig.on_clicked(on_savefig)
    orient_radio.on_clicked(on_orient)
    obj_radio.on_clicked(on_objsel)
    scene_state['sl_ox'].on_changed(on_obj_pos)
    scene_state['sl_oy'].on_changed(on_obj_pos)
    scene_state['sl_oz'].on_changed(on_obj_pos)
    scene_state['sl_os'].on_changed(on_obj_pos)
    scene_state['sl_rx'].on_changed(on_obj_pos)
    scene_state['sl_ry'].on_changed(on_obj_pos)
    scene_state['sl_rz'].on_changed(on_obj_pos)
    _full_update(0.0)
    plt.show()


def precompute(df_cal, df_raw, resting_positions):
    results = {}
    g = 9.81
    for finger in ('A', 'B', 'C'):
        motor     = finger_motor[finger]
        pos_col   = position_col[motor]
        sc        = sensor_cols[finger]
        resting   = resting_positions[finger]
        direction = finger_direction[finger]
        th1_arr, th2_arr, th3_arr, th4_arr, Fc_arr = [], [], [], [], []
        F2_arr,  F3_arr,  F4_arr                   = [], [], []

        for i in range(len(df_cal)):
            pos      = df_raw[pos_col].iloc[i] if pd.notna(df_raw[pos_col].iloc[i]) else resting
            raw_deg  = max(0.0, direction * (pos - resting))
            raw_deg  = min(raw_deg, MAX_MOTOR_DEG)
            DeltaPhi = np.deg2rad(raw_deg + PRETENSION_DEG) * PREDICTION_GAIN
            row_cal  = df_cal.iloc[i]
            # Sensor → force mapping (C1=base-proximal, C3=tip):
            #   L0 (palm-proximal) has NO sensor → F1 = 0
            #   L1 = C1 sensor (sc[0]) → F2
            #   L2 = C2 sensor (sc[1]) → F3
            #   L3 (tip) = C3 sensor (sc[2]) → F4
            F2_N = row_cal[sc[0]] / 1000 * g * FORCE_EFFECT   # C1 → L1
            F3_N = row_cal[sc[1]] / 1000 * g * FORCE_EFFECT   # C2 → L2
            F4_N = row_cal[sc[2]] / 1000 * g * FORCE_EFFECT   # C3 → L3 tip
            th1, th2, th3, th4, Fc = solve_angles(DeltaPhi, 0.0, F2_N, F3_N, F4_N)
            th1_arr.append(th1); th2_arr.append(th2)
            th3_arr.append(th3); th4_arr.append(th4)
            Fc_arr.append(Fc)
            F2_arr.append(F2_N); F3_arr.append(F3_N); F4_arr.append(F4_N)

        results[finger] = {
            'th1': np.array(th1_arr), 'th2': np.array(th2_arr),
            'th3': np.array(th3_arr), 'th4': np.array(th4_arr),
            'Fc':  np.array(Fc_arr),
            'F2':  np.array(F2_arr),  'F3':  np.array(F3_arr),
            'F4':  np.array(F4_arr),
        }
    return results


def smooth_precomp(precomp):
    sigma    = max(1, ANIM_SMOOTH_FRAMES / 2)
    smoothed = {}
    for finger, data in precomp.items():
        smoothed[finger] = {
            key: gaussian_filter1d(arr.astype(float), sigma=sigma)
            for key, arr in data.items()
        }
    return smoothed


def compute_axis_limits(precomp_smooth):
    all_x, all_y = [], []
    for finger, data in precomp_smooth.items():
        n = len(data['th1'])
        for i in range(n):
            pts = forward_kinematics_2d(
                data['th1'][i], data['th2'][i],
                data['th3'][i], data['th4'][i])
            # pts = (p0, pB, pJ1, p1, pJ2, p2, pJ3, p3, pJ4, p4, …)
            for p in pts[:10]:   # p0..p4 bound the shape
                all_x.append(p[0]); all_y.append(p[1])
    margin = 0.015
    return (min(all_x) - margin, max(all_x) + margin), \
           (min(all_y) - margin, max(all_y) + margin)


def launch_prediction_viewer(df_raw, df_cal_p3, df_cal_p2,
                             precomp_raw_p3, precomp_raw_p2,
                             baselines, resting_positions):
    from matplotlib.widgets import Button as _Button, Slider as _Slider, RadioButtons as _RadioButtons

    precomp_p3 = smooth_precomp(precomp_raw_p3)
    precomp_p2 = smooth_precomp(precomp_raw_p2)

    _init_mode = 'poly3' if _POLY3_COEFFS else 'poly2'
    active = {
        'df_cal':  df_cal_p3 if _POLY3_COEFFS else df_cal_p2,
        'precomp': precomp_p3 if _POLY3_COEFFS else precomp_p2,
        'mode':    _init_mode,
    }

    def _cal_label():
        return f"cal: {active['mode']}"

    timestamps = df_raw['timestamp'].values
    n          = len(timestamps)

    print('Computing fixed axis limits …')
    fixed_xlim, fixed_ylim = compute_axis_limits(active['precomp'])


    scene = {'z_sign': 1, 'object_name': 'None'}
    (fig3d, ax3d, ax_side_3d,
     finger_cols, tip_dots, info_txt_3d,
     title_txt_3d, _, scene_state) = build_3d_figure(
         z_sign=scene['z_sign'], object_name=scene['object_name'])
    title_txt_3d.set_text('3-Finger Compliant Gripper — Prediction (3-D)')


    orient_radio_3d = mwidgets.RadioButtons(
        scene_state['ax_orient'], ['Hand Up', 'Hand Down'],
        active=0, activecolor='#2060b8')
    for lbl in orient_radio_3d.labels:
        lbl.set_fontsize(8); lbl.set_fontfamily('monospace')

    obj_radio_3d = mwidgets.RadioButtons(
        scene_state['ax_objsel'], OBJECT_NAMES,
        active=0, activecolor='#c86010')
    for lbl in obj_radio_3d.labels:
        lbl.set_fontsize(7.5); lbl.set_fontfamily('monospace')

    def _rebuild_obj_pred():
        """Remove old object, add new one for prediction 3-D view."""
        zs = scene['z_sign']
        ax = scene_state['ax']
        if scene_state['obj_col'] is not None:
            try: scene_state['obj_col'].remove()
            except ValueError: pass
            scene_state['obj_col'] = None
        off = tuple(scene_state['obj_offset'])
        scl = scene_state['obj_scale']
        rot = tuple(scene_state['obj_rotation'])
        obj_faces, obj_fc = build_object(scene['object_name'], z_sign=zs,
                                         offset=off, scale=scl,
                                         rotation_deg=rot)
        if obj_faces is not None:
            col = Poly3DCollection(
                obj_faces, facecolors=obj_fc,
                edgecolors=[(0.25, 0.25, 0.25, 0.5)] * len(obj_faces),
                linewidth=0.3, alpha=0.92, zsort='average')
            ax.add_collection3d(col)
            scene_state['obj_col'] = col

    def _rebuild_body_pred():
        zs = scene['z_sign']
        bdx = FINGER_SEPARATION / 2 + bw / 2 + 0.006
        bdy = PALM_HALF_SPAN + 0.012
        bdz = 0.022
        body_z1 = zs * (-bdz - BODY_DROP)
        body_z2 = zs * (-2 * bdz - 0.005 - BODY_DROP)
        scene_state['body_cols'][0].set_verts(
            _aabb_faces(0, 0, body_z1, bdx, bdy, bdz))
        scene_state['body_cols'][1].set_verts(
            _aabb_faces(0, 0, body_z2, bdx + 0.007, bdy + 0.007, 0.005))
        H = (l + lj) * 4 + l0 + 0.018
        if zs > 0:
            ax3d.set_zlim(-0.070, H)
        else:
            ax3d.set_zlim(-H, 0.070)

    def on_orient_pred(label):
        scene['z_sign'] = 1 if label == 'Hand Up' else -1
        _rebuild_body_pred()
        _rebuild_obj_pred()
        draw_frame(state['idx'])

    def on_objsel_pred(label):
        scene['object_name'] = label
        _rebuild_obj_pred()
        fig3d.canvas.draw_idle()

    def on_obj_pos_pred(_):
        scene_state['obj_offset'][0] = scene_state['sl_ox'].val
        scene_state['obj_offset'][1] = scene_state['sl_oy'].val
        scene_state['obj_offset'][2] = scene_state['sl_oz'].val
        scene_state['obj_scale']        = scene_state['sl_os'].val
        scene_state['obj_rotation'][0]  = scene_state['sl_rx'].val
        scene_state['obj_rotation'][1]  = scene_state['sl_ry'].val
        scene_state['obj_rotation'][2]  = scene_state['sl_rz'].val
        _rebuild_obj_pred()
        fig3d.canvas.draw_idle()

    orient_radio_3d.on_clicked(on_orient_pred)
    obj_radio_3d.on_clicked(on_objsel_pred)
    scene_state['sl_ox'].on_changed(on_obj_pos_pred)
    scene_state['sl_oy'].on_changed(on_obj_pos_pred)
    scene_state['sl_oz'].on_changed(on_obj_pos_pred)
    scene_state['sl_os'].on_changed(on_obj_pos_pred)
    scene_state['sl_rx'].on_changed(on_obj_pos_pred)
    scene_state['sl_ry'].on_changed(on_obj_pos_pred)
    scene_state['sl_rz'].on_changed(on_obj_pos_pred)

    fig2d = plt.figure(figsize=(18, 10), facecolor='white')
    fig2d.suptitle('Shape-Prediction Replay',
                   fontsize=14, fontweight='bold', fontfamily='monospace')

    gs = fig2d.add_gridspec(3, 1,
                            left=0.33, right=0.99,
                            top=0.93, bottom=0.07,
                            hspace=0.38)
    ax_A = fig2d.add_subplot(gs[0])
    ax_B = fig2d.add_subplot(gs[1])
    ax_C = fig2d.add_subplot(gs[2])

   
    ax_info = fig2d.add_axes([0.01, 0.52, 0.29, 0.41])
    ax_stat = fig2d.add_axes([0.01, 0.32, 0.29, 0.05])
    ax_info.axis('off'); ax_stat.axis('off')

    from matplotlib.widgets import TextBox as _TextBox

    _ov_state = {'active': False}
    _ov_values = {}   
    _ov_boxes  = {}  

    ax_ov_toggle = fig2d.add_axes([0.02, 0.480, 0.12, 0.025])
    btn_ov_toggle = _Button(ax_ov_toggle, 'Override: OFF',
                            color='#eee', hovercolor='#ddd')
    btn_ov_toggle.label.set_fontfamily('monospace')
    btn_ov_toggle.label.set_fontsize(8)

    _ov_x0    = 0.08       
    _ov_dx    = 0.075       
    _ov_tb_w  = 0.060      
    _ov_tb_h  = 0.022   
    _ov_y0    = 0.450      
    _ov_dy    = 0.028    

    ax_ov_hdr = fig2d.add_axes([0.01, _ov_y0 + 0.003, 0.28, 0.018])
    ax_ov_hdr.axis('off')
    for ci, sname in enumerate(('S1 (prox)', 'S2 (mid)', 'S3 (dist)')):
        ax_ov_hdr.text((_ov_x0 - 0.01 + ci * _ov_dx) / 0.28, 0.5,
                        sname, fontsize=6.5, fontfamily='monospace',
                        color='#555', ha='left', va='center',
                        transform=ax_ov_hdr.transAxes)

    _sensor_order = [
        ('A', ('A1', 'A2', 'A3')),
        ('B', ('B1', 'B2', 'B3')),
        ('C', ('C1', 'C2', 'C3')),
    ]
    for ri, (flabel, sensors) in enumerate(_sensor_order):
        y_row = _ov_y0 - ri * _ov_dy
        # Row label
        ax_rl = fig2d.add_axes([0.02, y_row, 0.04, _ov_tb_h])
        ax_rl.axis('off')
        ax_rl.text(0.9, 0.5, f'{flabel}:', fontsize=8, fontfamily='monospace',
                   fontweight='bold', color=FINGER_CLR[flabel],
                   ha='right', va='center', transform=ax_rl.transAxes)
        for ci, sname in enumerate(sensors):
            _ov_values[sname] = 0.0
            ax_tb = fig2d.add_axes([_ov_x0 + ci * _ov_dx, y_row,
                                    _ov_tb_w, _ov_tb_h])
            tb = _TextBox(ax_tb, '', initial='0.0')
            tb.label.set_fontsize(7)
            tb.text_disp.set_fontfamily('monospace')
            tb.text_disp.set_fontsize(8)
            _ov_boxes[sname] = tb

            def _make_cb(key):
                def _on_submit(text):
                    try:
                        _ov_values[key] = float(text)
                    except ValueError:
                        _ov_values[key] = 0.0
                    if _ov_state['active']:
                        state['ema'] = {f: None for f in ('A', 'B', 'C')}
                        draw_frame(state['idx'])
                return _on_submit
            tb.on_submit(_make_cb(sname))

    def _on_ov_toggle(event):
        _ov_state['active'] = not _ov_state['active']
        label = 'Override: ON' if _ov_state['active'] else 'Override: OFF'
        btn_ov_toggle.label.set_text(label)
        ax_ov_toggle.set_facecolor('#c8ffc8' if _ov_state['active'] else '#eee')
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])
    btn_ov_toggle.on_clicked(_on_ov_toggle)

    finger_axes = {'A': ax_A, 'B': ax_B, 'C': ax_C}

    _span = l0 + 4*(lj + l) + 0.01
    for f, ax in finger_axes.items():
        ax.set_xlim(-0.005, _span)
        ax.set_ylim(-0.075, 0.075)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title(f'Finger {f}', fontsize=10, fontfamily='monospace',
                     color=FINGER_CLR[f], fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(labelsize=7)

    link_colors  = ['#2060b8', '#3080d8', '#60a8f0', '#90c8ff']
    joint_colors = ['#8B5E3C', '#9B6E4C', '#AB7E5C', '#BB8E6C']

    artists = {}
    for f, ax in finger_axes.items():
        base_ln, = ax.plot([], [], '-', linewidth=10, color='#333',
                           solid_capstyle='round', zorder=2)
        j1, = ax.plot([], [], linewidth=5, color=joint_colors[0],
                      solid_capstyle='butt', zorder=2)
        j2, = ax.plot([], [], linewidth=5, color=joint_colors[1],
                      solid_capstyle='butt', zorder=2)
        j3, = ax.plot([], [], linewidth=5, color=joint_colors[2],
                      solid_capstyle='butt', zorder=2)
        j4, = ax.plot([], [], linewidth=5, color=joint_colors[3],
                      solid_capstyle='butt', zorder=2)
        l1, = ax.plot([], [], linewidth=8, color=link_colors[0],
                      solid_capstyle='round', zorder=3)
        l2, = ax.plot([], [], linewidth=8, color=link_colors[1],
                      solid_capstyle='round', zorder=3)
        l3, = ax.plot([], [], linewidth=8, color=link_colors[2],
                      solid_capstyle='round', zorder=3)
        l4, = ax.plot([], [], linewidth=8, color=link_colors[3],
                      solid_capstyle='round', zorder=3)
        jdots, = ax.plot([], [], 'o', markersize=6,
                         markerfacecolor='#444', markeredgecolor='#222',
                         markeredgewidth=1.0, zorder=4)
        tip,   = ax.plot([], [], 'o', markersize=11,
                         markerfacecolor='#cc2020', markeredgecolor='#880000',
                         markeredgewidth=1.4, zorder=5)
        artists[f] = dict(base=base_ln, j1=j1, j2=j2, j3=j3, j4=j4,
                          l1=l1, l2=l2, l3=l3, l4=l4,
                          joints=jdots, tip=tip,
                          arrows=[None, None, None])

    info_text = ax_info.text(
        0.03, 0.99, '', transform=ax_info.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                  edgecolor='#999', linewidth=1.2))
    status_text = ax_stat.text(
        0.03, 0.75, 'Loading…', transform=ax_stat.transAxes,
        fontsize=9.5, fontfamily='monospace', color='darkorange', va='top')

    
    _LX  = 0.02   # left edge of all controls
    _LW  = 0.28   # width of all sliders
    _SH  = 0.024  # slider height
    _BH  = 0.030  # button height
    _BW  = 0.085  # button width
    _DY  = 0.040  # row pitch
    _FS  = 9      # font size for all slider labels/values

    def _sl2(ax_pos, label, lo, hi, val, color):
        ax = fig2d.add_axes([_LX, ax_pos, _LW, _SH])
        sl = _Slider(ax, label, lo, hi, valinit=val, color=color)
        for _w in (sl.label, sl.valtext):
            _w.set_fontfamily('monospace'); _w.set_fontsize(_FS)
        return sl

    _y0 = 0.330   # top slider bottom edge
    
    ax_slider = fig2d.add_axes([_LX, _y0, _LW, _SH])
    slider2d  = _Slider(ax_slider, 'Time [s]',
                        timestamps[0], timestamps[-1],
                        valinit=timestamps[0], color='steelblue')
    for _w in (slider2d.label, slider2d.valtext):
        _w.set_fontfamily('monospace'); _w.set_fontsize(_FS)

    
    sl_gain  = _sl2(_y0 - 1*_DY, 'Gain',             0.1,  3.0,  PREDICTION_GAIN,  '#c06010')

    sl_rest  = _sl2(_y0 - 2*_DY, 'Rest offset [°]', -60.0, 60.0, -7.0,             '#106080')

    sl_eeff  = _sl2(_y0 - 3*_DY, 'E_eff [MPa]',      0.5,  30.0, E_EFFECTIVE/1e6,  '#601060')

    sl_feff  = _sl2(_y0 - 4*_DY, 'Force Effect',     0.0,   1.0, FORCE_EFFECT,     '#10a040')

    sl_pret  = _sl2(_y0 - 5*_DY, 'Pretension [°]',   0.0,  60.0, PRETENSION_DEG,   '#8050a0')

    sl_ct    = _sl2(_y0 - 6*_DY, 'Tendon Compl.',     0.0,  0.05, C_TENDON,         '#a06050')

    sl_splay = _sl2(_y0 - 7*_DY, 'Splay A/B [°]',    0.0,  60.0, 0.0,              '#208080')

    tune = {'gain': PREDICTION_GAIN, 'rest_off': 0.0, 'e_eff': E_EFFECTIVE,
            'force_effect': FORCE_EFFECT, 'pretension': PRETENSION_DEG,
            'c_tendon': C_TENDON, 'splay_deg': 0.0}

    _both_modes = bool(_POLY3_COEFFS and _POLY2_COEFFS)
    _by = 0.008 
    ax_radio = fig2d.add_axes([_LX, _by, 0.10, 0.038])
    ax_radio.set_visible(_both_modes)
    radio_mode = _RadioButtons(ax_radio, ['poly3', 'poly2'],
                               active=0 if active['mode'] == 'poly3' else 1,
                               activecolor='#2060b8')
    for lbl in radio_mode.labels:
        lbl.set_fontsize(9); lbl.set_fontfamily('monospace')

   
    ax_play    = fig2d.add_axes([_LX + 0.12,           _by, _BW,        _BH])
    ax_export  = fig2d.add_axes([_LX + 0.12 + _BW + 0.005, _by, _BW + 0.01, _BH])
    ax_savefig = fig2d.add_axes([_LX + 0.12 + 2*(_BW + 0.005) + 0.01, _by,
                                  _BW, _BH])

    btn_play2   = _Button(ax_play,    '▶  Play',      color='#ddeeff', hovercolor='#bbddff')
    btn_export  = _Button(ax_export,  '⬇  Export',    color='#d4f5d4', hovercolor='#a8e8a8')
    btn_savefig = _Button(ax_savefig, '[~] Save Fig',  color='#fff3cc', hovercolor='#ffe88a')
    for _b in (btn_play2, btn_export, btn_savefig):
        _b.label.set_fontfamily('monospace'); _b.label.set_fontsize(9)

    
    export_rows = []

  
    cache = {
        'idx':              0,
        'smoothed_all':     {f: np.zeros(8) for f in ('A', 'B', 'C')},
        'forces_g_all':     {f: [0.0, 0.0, 0.0] for f in ('A', 'B', 'C')},
        'angles_per_finger': {f: (0.0, 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
                              for f in ('A', 'B', 'C')},
    }
    state = {'idx': 0, 'playing': False, 'timer': None,
             'ema': {f: None for f in ('A', 'B', 'C')}}
    EMA_A = 2.0 / (ANIM_SMOOTH_FRAMES + 1)

    def ema_update(finger, raw_vals):
        if state['ema'][finger] is None:
            state['ema'][finger] = np.array(raw_vals, dtype=float)
        else:
            state['ema'][finger] = (EMA_A * np.array(raw_vals, dtype=float)
                                    + (1 - EMA_A) * state['ema'][finger])
        return state['ema'][finger]

    
    def _advance_state(idx):
        smoothed_all = {}
        forces_g_all = {}
        for finger in ('A', 'B', 'C'):
            data    = active['precomp'][finger]
            sc      = sensor_cols[finger]
            row_cal = active['df_cal'].iloc[idx]
            motor_col = position_col[finger_motor[finger]]
            pos_raw   = df_raw[motor_col].iloc[idx] if pd.notna(df_raw[motor_col].iloc[idx]) else resting_positions[finger]
            raw_deg   = max(0.0, finger_direction[finger] * (
                            pos_raw - (resting_positions[finger] + tune['rest_off'])))
            raw_deg   = min(raw_deg, MAX_MOTOR_DEG)
            DeltaPhi  = np.deg2rad(raw_deg + tune['pretension']) * tune['gain']
            # Sensor → force mapping (C1=base-proximal, C3=tip, L0 has no sensor):
            # When manual override is active, use typed gram values instead of data
            if _ov_state['active']:
                C1_g = _ov_values[sc[0]]
                C2_g = _ov_values[sc[1]]
                C3_g = _ov_values[sc[2]]
            else:
                C1_g = float(row_cal[sc[0]])
                C2_g = float(row_cal[sc[1]])
                C3_g = float(row_cal[sc[2]])
            F2_N = C1_g / 1000 * 9.81 * tune['force_effect']  # C1→L1
            F3_N = C2_g / 1000 * 9.81 * tune['force_effect']  # C2→L2
            F4_N = C3_g / 1000 * 9.81 * tune['force_effect']  # C3→L3 tip
            _live_I = (1.0/12.0) * bw * JH**3
            _lk1, _lk2, _lk3, _lk4 = tune['e_eff'] * _live_I / lj
            th1, th2, th3, th4, Fc = solve_angles(DeltaPhi, 0.0, F2_N, F3_N, F4_N,
                                                   _lk_override=(_lk1, _lk2, _lk3, _lk4),
                                                   _ct_override=tune['c_tendon'])
            raw = np.array([th1, th2, th3, th4, Fc,
                            F2_N, F3_N, F4_N])
            smoothed_all[finger] = ema_update(finger, raw)
            if _ov_state['active']:
                forces_g_all[finger] = [_ov_values[sc[j]] for j in range(3)]
            else:
                forces_g_all[finger] = [float(row_cal[sc[j]]) for j in range(3)]

        angles_per_finger = {}
        for finger in ('A', 'B', 'C'):
            th1_f, th2_f, th3_f, th4_f, Fc_f, *_ = smoothed_all[finger]
            angles_per_finger[finger] = (
                th1_f,
                th2_f,
                th3_f,
                th4_f,
                Fc_f, forces_g_all[finger])

        # Update 3-D
        update_3d(fig3d, finger_cols, tip_dots, angles_per_finger,
                  z_sign=scene['z_sign'], splay_deg=tune['splay_deg'])

        # Build 3-D info text
        sel = 'A'   
        motor_pos = df_raw.iloc[idx][position_col[finger_motor[sel]]]
        resting   = resting_positions[sel]
        dphi_deg  = max(0.0, finger_direction[sel] * (motor_pos - resting))
        lines3d = [
            f't = {timestamps[idx]:.3f} s   frame {idx+1}/{n}',
            '',
            f'Motor   {motor_pos:.2f} deg',
            f'ΔΦ      {dphi_deg:.2f} deg',
            '',
            '    J4     J3     J2     J1    Σ      Fc',
            '  ────────────────────────────────────────',
        ]
        for f in ('A', 'B', 'C'):
            th1, th2, th3, th4, Fc, _ = angles_per_finger[f]
            s = th1 + th2 + th3 + th4
            lines3d.append(
                f'{f}  {np.rad2deg(th1):5.1f}  {np.rad2deg(th2):5.1f}'
                f'  {np.rad2deg(th3):5.1f}  {np.rad2deg(th4):5.1f}'
                f'  {np.rad2deg(s):5.1f}°  {Fc:.4f} N')
        lines3d += ['', 'Forces [g]  (J1 J2 J3)', '  ─────────────────────────────────']
        for f in ('A', 'B', 'C'):
            fg = forces_g_all[f]
            lines3d.append(f'{f}  [{fg[0]:5.1f}  {fg[1]:5.1f}  {fg[2]:5.1f}]')
        info_txt_3d.set_text('\n'.join(lines3d))

        cache['idx']               = idx
        cache['smoothed_all']      = smoothed_all
        cache['forces_g_all']      = forces_g_all
        cache['angles_per_finger'] = angles_per_finger

    
        _row = {
            'timestamp':     round(float(timestamps[idx]), 3),
            'frame':         idx + 1,
            'motor_pos_deg': round(float(df_raw.iloc[idx][position_col[finger_motor['A']]]), 3),
            'gain':          round(tune['gain'], 4),
            'rest_off_deg':  round(tune['rest_off'], 3),
            'E_eff_MPa':     round(tune['e_eff'] / 1e6, 2),
            'force_effect':  round(tune['force_effect'], 4),
            'pretension_deg': round(tune['pretension'], 2),
            'c_tendon':      round(tune['c_tendon'], 5),
            'cal_mode':      active['mode'],
            'force_override': _ov_state['active'],
        }
        for _f in ('A', 'B', 'C'):
            _sm = smoothed_all[_f]
            _fg = forces_g_all[_f]
            _row[f'th1_{_f}_deg'] = round(float(np.rad2deg(_sm[0])), 3)
            _row[f'th2_{_f}_deg'] = round(float(np.rad2deg(_sm[1])), 3)
            _row[f'th3_{_f}_deg'] = round(float(np.rad2deg(_sm[2])), 3)
            _row[f'th4_{_f}_deg'] = round(float(np.rad2deg(_sm[3])), 3)
            _row[f'th_sum_{_f}_deg'] = round(
                float(np.rad2deg(_sm[0]+_sm[1]+_sm[2]+_sm[3])), 3)
            _row[f'Fc_{_f}_N']  = round(float(_sm[4]), 5)
            _row[f'F1_{_f}_g']  = round(_fg[0], 2)
            _row[f'F2_{_f}_g']  = round(_fg[1], 2)
            _row[f'F3_{_f}_g']  = round(_fg[2], 2)
            for _s in sensor_cols[_f]:
                _row[f'{_s}_raw'] = int(df_raw.iloc[idx].get(_s, 0))
        export_rows.append(_row)

    def _redraw_2d_only():
        idx = cache['idx']

        for f, ax in finger_axes.items():
            _sm = cache['smoothed_all'][f]
            th1 = _sm[0]
            th2 = _sm[1]
            th3 = _sm[2]
            th4 = _sm[3]
            Fc  = _sm[4]
            g_vals = cache['forces_g_all'][f]
            art    = artists[f]

            (p0, pB, pJ1, p1,
             pJ2, p2, pJ3, p3,
             pJ4, p4,
             pm1, pm2, pm3, pm4) = forward_kinematics_2d(th1, th2, th3, th4)

            art['base'].set_data([p0[0],  pB[0]],  [p0[1],  pB[1]])
            art['j1'].set_data(  [pB[0],  pJ1[0]], [pB[1],  pJ1[1]])
            art['j2'].set_data(  [p1[0],  pJ2[0]], [p1[1],  pJ2[1]])
            art['j3'].set_data(  [p2[0],  pJ3[0]], [p2[1],  pJ3[1]])
            art['j4'].set_data(  [p3[0],  pJ4[0]], [p3[1],  pJ4[1]])
            art['l1'].set_data(  [pJ1[0], p1[0]],  [pJ1[1], p1[1]])
            art['l2'].set_data(  [pJ2[0], p2[0]],  [pJ2[1], p2[1]])
            art['l3'].set_data(  [pJ3[0], p3[0]],  [pJ3[1], p3[1]])
            art['l4'].set_data(  [pJ4[0], p4[0]],  [pJ4[1], p4[1]])
            art['joints'].set_data(
                [pB[0], pJ1[0], p1[0], pJ2[0], p2[0], pJ3[0], p3[0], pJ4[0]],
                [pB[1], pJ1[1], p1[1], pJ2[1], p2[1], pJ3[1], p3[1], pJ4[1]])
            art['tip'].set_data([p4[0]], [p4[1]])

            # Force arrows — C1 on L1(pm2), C2 on L2(pm3), C3/tip on L3(pm4)
            n2 = np.array([-np.sin(th1+th2),           np.cos(th1+th2)])
            n3 = np.array([-np.sin(th1+th2+th3),       np.cos(th1+th2+th3)])
            n4 = np.array([-np.sin(th1+th2+th3+th4),   np.cos(th1+th2+th3+th4)])
            for q in art['arrows']:
                if q is not None: q.remove()
            for i, (pt, nv, gv) in enumerate(
                    zip([pm2, pm3, pm4], [n2, n3, n4], g_vals)):
                if gv > 0.5:
                    alen = gv * ARROW_G_SCALE
                    art['arrows'][i] = ax.quiver(
                        pt[0], pt[1], -alen*nv[0], -alen*nv[1],
                        color='#cc1111', linewidth=2,
                        scale=1, scale_units='xy', angles='xy', zorder=6)
                else:
                    art['arrows'][i] = None

        f   = 'A'   # show finger A as primary detail; all angles shown in 3D
        sc  = sensor_cols[f]
        _sm_f = cache['smoothed_all'][f]
        th1 = _sm_f[0]
        th2 = _sm_f[1]
        th3 = _sm_f[2]
        th4 = _sm_f[3]
        Fc  = _sm_f[4]
        motor     = finger_motor[f]
        row_raw   = df_raw.iloc[idx]
        row_cal   = active['df_cal'].iloc[idx]
        pos       = row_raw[position_col[motor]]
        resting   = resting_positions[f]
        dphi_deg  = max(0.0, finger_direction[f] * (pos - resting))
        bl        = [baselines[s] for s in sc]
        raw_vals  = [row_raw[s]  for s in sc]
        cal_vals  = [row_cal[s]  for s in sc]
        abs_d     = [abs(bl[j] - raw_vals[j]) for j in range(3)]

        _eff_resting = resting + tune['rest_off']
        dphi_deg  = max(0.0, finger_direction[f] * (pos - _eff_resting))
        lines_info = [
            f'Motor: {pos:.1f}°  (rest: {_eff_resting:.1f}°)',
            f'ΔΦ: {dphi_deg:.2f}°  ({dphi_deg*np.pi/180:.3f} rad)',
            f'Gain: {tune["gain"]:.2f}  E_eff: {tune["e_eff"]/1e6:.1f}MPa',
            f'Force Effect: {tune["force_effect"]:.2f}  Pretension: {tune["pretension"]:.1f}°',
            f'Cal: {_cal_label()}',
        ]
        if _ov_state['active']:
            lines_info.append('⚠ FORCE OVERRIDE ACTIVE')
        lines_info += [
            '',
            '    J1    J2    J3    J4   Σ',
            '  ──────────────────────────',
        ]
        for fi in ('A', 'B', 'C'):
            _sm_fi = cache['smoothed_all'][fi]
            t1 = _sm_fi[0]
            t2 = _sm_fi[1]
            t3 = _sm_fi[2]
            t4 = _sm_fi[3]
            s = t1 + t2 + t3 + t4
            lines_info.append(
                f'{fi} {np.rad2deg(t1):5.1f} {np.rad2deg(t2):5.1f}'
                f' {np.rad2deg(t3):5.1f} {np.rad2deg(t4):5.1f}'
                f' {np.rad2deg(s):5.1f}°')
        lines_info += ['', 'Forces [g]  (J1 J2 J3)', '  ─────────────────────']
        for fi in ('A', 'B', 'C'):
            fg = cache['forces_g_all'][fi]
            lines_info.append(f'{fi} [{fg[0]:5.1f} {fg[1]:5.1f} {fg[2]:5.1f}]')
        lines_info += ['', 'Raw ADC readings', '  ─────────────────────',
                       '    S1    S2    S3']
        for fi in ('A', 'B', 'C'):
            s3, s2, s1 = sensor_cols[fi]
            r3 = int(df_raw.iloc[idx][s3])
            r2 = int(df_raw.iloc[idx][s2])
            r1 = int(df_raw.iloc[idx][s1])
            b3 = baselines[s3]; b2 = baselines[s2]; b1 = baselines[s1]
            d3 = r3 - b3;       d2 = r2 - b2;       d1 = r1 - b1
            lines_info.append(
                f'{fi} {r3:5d} {r2:5d} {r1:5d}'
                f'  Δ({d3:+.0f},{d2:+.0f},{d1:+.0f})')
        info_text.set_text('\n'.join(lines_info))
        status_text.set_text(
            f't = {timestamps[idx]:.3f} s   frame {idx+1}/{n}   {_cal_label()}')
        status_text.set_color('#226622')
        fig2d.canvas.draw_idle()


    def draw_frame(idx):
        _advance_state(idx)
        _redraw_2d_only()

    def on_slider(val):
        idx = min(int(np.searchsorted(timestamps, slider2d.val)), n - 1)
        state['idx'] = idx
        draw_frame(idx)

    def step_forward():
        idx = (state['idx'] + 1) % n
        state['idx'] = idx
        slider2d.eventson = False
        slider2d.set_val(timestamps[idx])
        slider2d.eventson = True
        draw_frame(idx)

    def on_play(event):
        if not state['playing']:
            state['playing'] = True
            btn_play2.label.set_text('||  Pause')
            state['timer'] = fig2d.canvas.new_timer(interval=ANIM_INTERVAL_MS)
            state['timer'].add_callback(step_forward)
            state['timer'].start()
        else:
            state['playing'] = False
            btn_play2.label.set_text('>>  Play')
            if state['timer']:
                state['timer'].stop(); state['timer'] = None
        fig2d.canvas.draw_idle()

    def on_mode_toggle(label):
        active['mode']    = label
        active['df_cal']  = df_cal_p3  if label == 'poly3' else df_cal_p2
        active['precomp'] = precomp_p3 if label == 'poly3' else precomp_p2
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_gain(val):
        tune['gain'] = float(val)
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_rest(val):
        tune['rest_off'] = float(val)
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_eeff(val):
        tune['e_eff'] = float(val) * 1e6
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_feff(val):
        tune['force_effect'] = float(val)
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_pret(val):
        tune['pretension'] = float(val)
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_ct(val):
        tune['c_tendon'] = float(val)
        state['ema'] = {f: None for f in ('A', 'B', 'C')}
        draw_frame(state['idx'])

    def on_splay(val):
        tune['splay_deg'] = float(val)
        draw_frame(state['idx'])

    def on_export(event):
        """Save logged frames to Excel + a JSON settings sidecar."""
        rows = list(export_rows)
        if not rows:
            print('  No data to export (play through some frames first).')
            return
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'finger_sim_replay_{ts}'
        print(f'\n  Exporting {len(rows)} frames …')
        print(f'  Default filename: {default_name}.xlsx')
        print(f'  Save directory  : {SAVE_DIR}')
        try:
            user_input = input('  Enter filename [blank=default, "skip"=cancel]: ').strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ''
        if user_input.lower() == 'skip':
            print('  Export cancelled.')
            return
        name = user_input if user_input else default_name
        if name.lower().endswith('.xlsx'):
            name = name[:-5]
        path = os.path.join(SAVE_DIR, f'{name}.xlsx')
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            pd.DataFrame(rows).to_excel(path, index=False)
            print(f'  ✓ Saved {len(rows)} rows → {path}')
        except Exception as e:
            print(f'  Save failed: {e}')
            fallback = os.path.join(SAVE_DIR, f'{default_name}.xlsx')
            try:
                pd.DataFrame(rows).to_excel(fallback, index=False)
                print(f'  ✓ Fallback saved → {fallback}')
            except Exception as e2:
                print(f'  Fallback also failed: {e2}')
            name = default_name  # use fallback stem for settings file

        import json
        settings = {
            'saved_at':       datetime.now().isoformat(),
            'cal_mode':       active['mode'],
            'gain':           round(tune['gain'], 4),
            'rest_off_deg':   round(tune['rest_off'], 3),
            'E_eff_MPa':      round(tune['e_eff'] / 1e6, 2),
            'force_effect':   round(tune['force_effect'], 4),
            'pretension_deg': round(tune['pretension'], 2),
            'c_tendon':       round(tune['c_tendon'], 5),
            'EFFECTIVE_RP':   EFFECTIVE_RP,
            'PREDICTION_GAIN_default': PREDICTION_GAIN,
            'FORCE_EFFECT_default': FORCE_EFFECT,
            'PRETENSION_DEG_default': PRETENSION_DEG,
            'C_TENDON_default': C_TENDON,
            'resting_positions': {k: round(v, 3)
                                  for k, v in resting_positions.items()},
        }
        jpath = os.path.join(SAVE_DIR, f'{name}_settings.json')
        try:
            with open(jpath, 'w') as jf:
                json.dump(settings, jf, indent=2)
            print(f'  ✓ Settings saved → {jpath}')
            print(f'     gain={settings["gain"]}  rest_off={settings["rest_off_deg"]}°'
                  f'  E_eff={settings["E_eff_MPa"]} MPa  cal={settings["cal_mode"]}')
        except Exception as je:
            print(f'  Settings save failed: {je}')

    def on_savefig(event):
        """Save 2-D figure as thesis-quality PNG + PDF."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(SAVE_DIR, exist_ok=True)
        for ext in ('png', 'pdf'):
            p = os.path.join(SAVE_DIR, f'gripper_2d_{ts}.{ext}')
            fig2d.savefig(p, dpi=300, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
        print(f'  ✓ Figure saved → {SAVE_DIR}  (PNG + PDF)')

    slider2d.on_changed(on_slider)
    sl_gain.on_changed(on_gain)
    sl_rest.on_changed(on_rest)
    sl_eeff.on_changed(on_eeff)
    sl_feff.on_changed(on_feff)
    sl_pret.on_changed(on_pret)
    sl_ct.on_changed(on_ct)
    sl_splay.on_changed(on_splay)
    btn_play2.on_clicked(on_play)
    btn_export.on_clicked(on_export)
    btn_savefig.on_clicked(on_savefig)
    if _both_modes:
        radio_mode.on_clicked(on_mode_toggle)
    draw_frame(0)
    plt.show()


def _pick_file(title, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path or None
    except Exception:
        return None


if __name__ == '__main__':
    xl = [('Excel files', '*.xlsx *.xls')]

    print('\n=== Step 1/2 — Calibration file (optional) ===')
    print('Select a calibration Excel file, or cancel for fallback linear scale.')
    load_calibration(
        _pick_file('Select calibration Excel file (cancel to skip)', xl))

    print('\n=== Step 2/2 — Sensor data file (optional) ===')
    print('Select a sensor-data Excel file, or cancel for simulation only.')
    data_path = _pick_file(
        'Select sensor data Excel file (cancel for simulation only)', xl)

    if not data_path:
        print('\nNo data file — launching simulation mode.')
        launch_simulation()
    else:
        print(f'\nLoading: {data_path}')
        df_raw = pd.read_excel(data_path)
        print(f'  {len(df_raw)} rows, {len(df_raw.columns)} columns')

        for motor, col in position_col.items():
            if col in df_raw.columns:
                print(f'  {motor}: min={df_raw[col].min():.1f}  '
                      f'max={df_raw[col].max():.1f}  '
                      f'median={df_raw[col].median():.1f} deg')

        print('\nDetecting resting positions …')
        resting_positions = compute_resting_positions(df_raw)
        for finger, pos in resting_positions.items():
            motor = finger_motor[finger]
            print(f'  Finger {finger} ({motor}  dir={finger_direction[finger]:+d}  '
                  f'pct={finger_resting_percentile[finger]:.2f}): '
                  f'resting = {pos:.2f} deg')

        print('\nCalibrating sensors — poly3 …')
        CAL_POLY_MODE = 'poly3'
        df_cal_p3, baselines = calibrate_sensors(df_raw)
        for s, bv in baselines.items():
            print(f'    {s}: {bv:.1f}')

        print('Calibrating sensors — poly2 …')
        CAL_POLY_MODE = 'poly2'
        df_cal_p2, _ = calibrate_sensors(df_raw)

        print('\nPrecomputing angles — poly3 (this may take a moment) …')
        CAL_POLY_MODE = 'poly3'
        precomp_p3 = precompute(df_cal_p3, df_raw, resting_positions)

        print('Precomputing angles — poly2 …')
        CAL_POLY_MODE = 'poly2'
        precomp_p2 = precompute(df_cal_p2, df_raw, resting_positions)

        CAL_POLY_MODE = 'poly3' if _POLY3_COEFFS else 'poly2'
        print('Done — launching prediction viewer.')
        launch_prediction_viewer(df_raw, df_cal_p3, df_cal_p2,
                                 precomp_p3, precomp_p2,
                                 baselines, resting_positions)