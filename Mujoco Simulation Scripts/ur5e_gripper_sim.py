"""
ur5e_gripper_sim.py — UR5e + 3-Finger Compliant Gripper  Pick & Place

"""

import time, os, sys, threading
import numpy as np
import mujoco
import tkinter as tk
from tkinter import ttk

#    Windows DPI fix                                                   
try:
    import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

 
#  PHYSICAL CONSTANTS
 
EFFECTIVE_RP  = 0.007
H_MOMENT_ARM  = 0.0065
MAX_MOTOR_DEG = 450.0
CALIB_MAX_DEG = 700.0

E  = 67e6;  bw = 10e-3;  lj = 15e-3
JH = np.array([3e-3, 3.25e-3, 3.25e-3, 4e-3])
_I = (1.0/12.0) * bw * JH**3
HOLLOW_FACTOR = 0.45
K_JOINTS = E * _I / lj * HOLLOW_FACTOR

DEFLECTION_GAIN = 300.0
CONTACT_SCALE   = 0.14
FINGERS         = ['A', 'B', 'C']

 
#  ARM CONSTANTS
 
ARM_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
ARM_ACT_NAMES = [
    'shoulder_pan', 'shoulder_lift', 'elbow',
    'wrist_1', 'wrist_2', 'wrist_3',
]
N_ARM = 6

CART_STEP  = 0.003
ROT_STEP   = 0.04
IK_LAMBDA  = 0.015     # Slightly lower damping for faster convergence
IK_POS_GAIN = 12.0     # Higher position gain for faster tracking

TABLE_Z = 0.400   # table surface z
TRAY_Z  = 0.503   # pick pedestal top surface z
PLACE_BOX_Z = 0.403  # place box floor z (at table level)

 
#  TRAY POSITIONS
 
PICK_TRAY_XY  = np.array([-0.22, 0.35])   # green tray centre
PLACE_TRAY_XY = np.array([ 0.52, 0.35])   # blue tray centre

 
#  OBJECTS
 
OBJ_NAMES = [
    'obj_sphere', 'obj_paint_tube', 'obj_screwdriver',
    'obj_tape_roll', 'obj_paper_roll',
]
OBJ_DISPLAY = {
    'obj_sphere':      'Sphere',
    'obj_paint_tube':  'Paint Tube',
    'obj_screwdriver': 'Screwdriver',
    'obj_tape_roll':   'Tape Roll',
    'obj_paper_roll':  'Paper Roll',
}

QUAT_CYL_HORIZ = np.array([0.7071068, 0.0, 0.7071068, 0.0])
QUAT_IDENTITY  = np.array([1.0, 0.0, 0.0, 0.0])

# Resting z-height above TRAY_Z for each object (half-height of collision geom)
OBJ_Z_OFFSET = {
    'obj_sphere':      0.016,
    'obj_paint_tube':  0.017,
    'obj_screwdriver': 0.0127,
    'obj_tape_roll':   0.0254,
    'obj_paper_roll':  0.1016,   # Standing upright: half-height of cylinder
}
OBJ_QUAT = {
    'obj_sphere':      QUAT_IDENTITY,
    'obj_paint_tube':  QUAT_CYL_HORIZ,
    'obj_screwdriver': QUAT_CYL_HORIZ,
    'obj_tape_roll':   QUAT_IDENTITY,
    'obj_paper_roll':  QUAT_IDENTITY,   # Standing upright (vertical)
}

#    Approach style                                                     
# Objects that need a lateral (side) approach so the fingers wrap around
# the cylinder from the side, matching the real experiment.
SIDE_APPROACH_OBJS = {'obj_paper_roll', 'obj_tape_roll', 'obj_paint_tube', 'obj_screwdriver'}

# Top-down approach parameters (sphere / compact objects)
GRASP_HOVER   = 0.25   # pre-grasp hover height above table (above pedestal)
GRASP_DESCEND = 0.12   # grasp approach: TCP z above table (at pedestal level)

# Side approach parameters (cylindrical / elongated objects)
#   Three-phase approach: (1) high above & behind, (2) descend to grasp Z,
#   (3) linear forward to grasp point.  This prevents the forearm from
#   swinging through the object space.
SIDE_APPROACH_OFFSET = 0.16   # stand-off distance in -Y before approaching
SIDE_RETREAT_OFFSET  = 0.16   # retreat distance in -Y after dropping
SIDE_HOVER_HEIGHT    = 0.35   # hover height above TABLE_Z before descending
# (must clear tallest object on raised tray: paper roll top ≈ 0.71m)

# TCP offset along the approach axis (world +Y for side approach).
# The TCP site is near the gripper base; fingers extend ~9cm further.
# By placing the TCP slightly behind the object centre the finger curl
# zone wraps around the object instead of overshooting.
TCP_APPROACH_OFFSET  = 0.025  # TCP positioned 2.5cm behind object centre

# Safety: never command the TCP below this Z (table surface + small margin)
TCP_Z_MIN = TABLE_Z + 0.025

LIFT_HEIGHT    = 0.38   # lift height above table (clear raised tray + object)
TRANSIT_HEIGHT = 0.42   # clearance height during transit

#    Gripper orientation targets                                       
# Rotation matrices where columns = body axes expressed in world frame.
# 
# The gripper_base is mounted with quat="0.7071 -0.7071 0 0" (-90° X rotation).
# This means gripper's Z axis (finger direction) = wrist's -Y axis.
# 
# These matrices specify the DESIRED orientation of gripper_base in world frame.
# The IK will move the arm joints to achieve this orientation.

# Pointing DOWN (home / top-down approach): 
#   - x_body = world +X
#   - y_body = world -Y  
#   - z_body = world -Z (fingers pointing down)
ROT_GRIPPER_DOWN = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

# HORIZONTAL approach from -Y toward +Y (side grasp):
#   - z_body (fingers) = world +Y (toward object)
#   - x_body = world +X (horizontal)
#   - y_body = world -Z (palm faces down)
# Finger curl plane: world XY → wraps around vertical cylinder (Z-axis) well
ROT_GRIPPER_SIDE_Y = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=float)

# PERPENDICULAR approach from -Y toward +Y, rotated 90° around approach axis:
#   - z_body (fingers) = world +Y (toward object)  — same approach direction
#   - x_body = world -Z (now vertical, not horizontal)
#   - y_body = world -X
# This rotates the finger curl plane 90° so fingers wrap perpendicular to
# a horizontal cylinder's axis. Essential for grasping cylinders whose long
# axis runs left-right (world X) while approaching from the side.
ROT_GRIPPER_SIDE_Y_PERP = np.array([
    [ 0, -1,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
], dtype=float)

# Orientation presets for each object — used when auto-run picks the best
# default orientation for the selected object.
OBJ_DEFAULT_ORI = {
    'obj_sphere':      'Down',
    'obj_paint_tube':  'Side-Perp',
    'obj_screwdriver': 'Side-Perp',
    'obj_tape_roll':   'Side-Perp',
    'obj_paper_roll':  'Side-Perp',
}

ORI_MATRICES = {
    'Down':      ROT_GRIPPER_DOWN,
    'Side':      ROT_GRIPPER_SIDE_Y,
    'Side-Perp': ROT_GRIPPER_SIDE_Y_PERP,
}

def obj_pick_pos(obj_name):
    """Return XYZ centre of object on PICK pedestal."""
    z = TRAY_Z + OBJ_Z_OFFSET[obj_name]
    return np.array([PICK_TRAY_XY[0], PICK_TRAY_XY[1], z])

def obj_place_pos(obj_name):
    """Return XYZ drop position ABOVE the PLACE box.
    Drop from well above the box walls (wall top = 0.466m).
    """
    z = 0.52  # drop height: above box walls, object falls in
    return np.array([PLACE_TRAY_XY[0], PLACE_TRAY_XY[1], z])

 
#  PICK & PLACE PHASE SEQUENCE
 
PHASES = [
    'HOME',
    'PRE_PICK',
    'PICK',
    'GRASP',
    'LIFT',
    'TRANSIT',
    'PRE_PLACE',
    'PLACE',
    'RELEASE',
    'RETURN',
]

PHASE_LABELS = {
    'HOME':      '🏠 Home',
    'PRE_PICK':  '⬆ Above/Behind',
    'PICK':      '⬇ Descend',
    'GRASP':     '➡✊ Approach+Grasp',
    'LIFT':      '⬆ Lift',
    'TRANSIT':   '➡ Over Box',
    'PRE_PLACE': '⬇ Drop Point',
    'PLACE':     '✋ Release',
    'RELEASE':   '⬆ Clear',
    'RETURN':    '↩ Return',
}

PHASE_COLORS = {
    'HOME':      '#4a90d9',
    'PRE_PICK':  '#7cb87c',
    'PICK':      '#5aad5a',
    'GRASP':     '#e8a020',
    'LIFT':      '#d4a020',
    'TRANSIT':   '#a070c0',
    'PRE_PLACE': '#5090c8',
    'PLACE':     '#808080',
    'RELEASE':   '#606090',
    'RETURN':    '#4a90d9',
}

 
#  LOOKUP TABLES
 
_jnt_qadr     = {}
_ten_ids      = {}
_act_pos      = {}
_act_frc      = {}
_act_splay    = {}
_touch_sensor = {}
_geom_to_seg  = {}
_obj_jnt_adr  = {}
_obj_jnt_vadr = {}
_arm_jnt_ids  = []
_arm_jnt_qadr = []
_arm_act_ids  = []
_arm_dof_ids  = []
_tcp_body_id  = -1
_free_angle_table = {}

# Collision avoidance lookups
_arm_geom_ids    = set()   # geom IDs belonging to arm links (not gripper/fingers)
_scene_geom_ids  = set()   # geom IDs that are static obstacles (table, pedestals, trays)


def _build_lookups(model):
    global _tcp_body_id
    _arm_jnt_ids.clear(); _arm_jnt_qadr.clear()
    _arm_act_ids.clear(); _arm_dof_ids.clear()
    for jname, aname in zip(ARM_JOINT_NAMES, ARM_ACT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        _arm_jnt_ids.append(jid)
        _arm_jnt_qadr.append(model.jnt_qposadr[jid])
        _arm_dof_ids.append(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        _arm_act_ids.append(aid)
    _tcp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')
    for f in FINGERS:
        for j in range(4):
            name = f'{f}_j{j}'
            jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            _jnt_qadr[name] = model.jnt_qposadr[jid]
        _ten_ids[f'tendon_{f}'] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TENDON, f'tendon_{f}')
        _act_pos[f] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_pos')
        _act_frc[f] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_frc')
        splay_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act_{f}_splay')
        if splay_aid >= 0:
            _act_splay[f] = splay_aid
        for gname in (f'{f}_palm_geom', f'{f}_j0_geom',
                      f'{f}_l0_geom', f'{f}_l0_lip_a', f'{f}_l0_lip_b',
                      f'{f}_l0_lip_c', f'{f}_l0_lip_d'):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                _geom_to_seg[gid] = (f, 0)
        for j in range(1, 4):
            tname = f'{f}_touch{j}'
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, tname)
            if sid >= 0:
                _touch_sensor[tname] = model.sensor_adr[sid]
            for gname in (f'{f}_l{j}_geom', f'{f}_j{j}_geom'):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    _geom_to_seg[gid] = (f, j)
            for suffix in ('lip_a', 'lip_b', 'lip_c', 'lip_d'):
                lgname = f'{f}_l{j}_{suffix}'
                lgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, lgname)
                if lgid >= 0:
                    _geom_to_seg[lgid] = (f, j)
    for oname in OBJ_NAMES:
        jname = oname + '_jnt'
        jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        _obj_jnt_adr[oname]  = model.jnt_qposadr[jid]
        _obj_jnt_vadr[oname] = model.jnt_dofadr[jid]
    print(f"  Lookups: {len(_arm_jnt_ids)} arm joints, "
          f"{len(_geom_to_seg)} finger geoms, {len(OBJ_NAMES)} objects")

    #    Build collision avoidance geom sets                       
    _arm_geom_ids.clear()
    _scene_geom_ids.clear()

    # Arm bodies whose collision geoms we want to protect
    arm_body_names = ['base', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                      'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'gripper_base']
    arm_body_ids = set()
    for bname in arm_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            arm_body_ids.add(bid)

    # Scene obstacle geom names (table, pedestals, tray/box walls)
    scene_geom_names = [
        'table_top', 'pick_pedestal', 'pick_pedestal_top',
        'place_box_floor', 'place_box_w_front', 'place_box_w_back',
        'place_box_w_left', 'place_box_w_right', 'floor',
    ]

    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        if body_id in arm_body_ids:
            _arm_geom_ids.add(gid)
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname and gname in scene_geom_names:
            _scene_geom_ids.add(gid)

    print(f"  Collision avoidance: {len(_arm_geom_ids)} arm geoms, "
          f"{len(_scene_geom_ids)} scene obstacle geoms")


 
#  COLLISION AVOIDANCE
 
COLLISION_REPULSE_GAIN = 0.08   # how hard to push TCP away per contact
COLLISION_REPULSE_MAX  = 0.015  # max displacement per frame (metres)

def check_arm_collisions(model, data, arm_cart_target):
    """Scan contacts for arm-vs-scene collisions.
    If found, push arm_cart_target away from the collision.
    Returns number of arm-scene contacts detected.
    """
    repulsion = np.zeros(3)
    n_contacts = 0

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        arm_hit = scene_hit = False

        if g1 in _arm_geom_ids and g2 in _scene_geom_ids:
            arm_hit = True; scene_hit = True
        elif g2 in _arm_geom_ids and g1 in _scene_geom_ids:
            arm_hit = True; scene_hit = True

        if not (arm_hit and scene_hit):
            continue

        n_contacts += 1
        # Contact frame: first 3 entries of c.frame are the contact normal
        normal = np.array([c.frame[0], c.frame[1], c.frame[2]])
        penetration = max(0.0, -c.dist)  # dist < 0 means penetrating

        # Push TCP in the direction of the contact normal, scaled by depth
        push = normal * (penetration + 0.005) * COLLISION_REPULSE_GAIN
        # Bias upward: collisions almost always mean the arm is too low
        push[2] = max(push[2], penetration * 0.02)
        repulsion += push

    if n_contacts > 0:
        # Clamp total repulsion magnitude
        mag = np.linalg.norm(repulsion)
        if mag > COLLISION_REPULSE_MAX:
            repulsion *= COLLISION_REPULSE_MAX / mag
        arm_cart_target += repulsion

    return n_contacts


 
#  CALIBRATION
 
def calibrate_free_angles(model, data):
    global _free_angle_table
    print("  Calibrating free-air angles …", end='', flush=True)
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    for oname in OBJ_NAMES:
        qa = _obj_jnt_adr[oname]
        data.qpos[qa:qa+3] = [5.0, 5.0, -5.0]
        data.qvel[_obj_jnt_vadr[oname]:_obj_jnt_vadr[oname]+6] = 0.0
    for i, aid in enumerate(_arm_act_ids):
        data.ctrl[aid] = data.qpos[_arm_jnt_qadr[i]]
    for f in FINGERS:
        if f in _act_splay:
            data.ctrl[_act_splay[f]] = 0.0
    for pct in range(0, 101):
        dphi   = np.deg2rad(pct / 100.0 * CALIB_MAX_DEG)
        target = EFFECTIVE_RP * dphi
        for f in FINGERS:
            data.ctrl[_act_pos[f]] = target
            data.ctrl[_act_frc[f]] = 0.0
        for _ in range(400):
            mujoco.mj_step(model, data)
        angles = np.zeros(12)
        idx = 0
        for f in FINGERS:
            for j in range(4):
                angles[idx] = data.qpos[_jnt_qadr[f'{f}_j{j}']]
                idx += 1
        _free_angle_table[pct] = angles
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl
    mujoco.mj_forward(model, data)
    print(f" done ({len(_free_angle_table)} points)")


def _get_free_angles(closure_pct, max_motor_deg=None):
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


 
#  FORCE ESTIMATION
 
def get_segment_forces(model, data):
    forces = {(f, j): 0.0 for f in FINGERS for j in range(4)}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)
        fn   = abs(wrench[0])
        ft   = np.linalg.norm(wrench[1:3])
        g    = np.sqrt(fn**2 + ft**2) / 9.81 * 1000.0
        seg1 = _geom_to_seg.get(c.geom1)
        seg2 = _geom_to_seg.get(c.geom2)
        if seg1 and seg2:
            forces[seg1] += g * 0.5
            forces[seg2] += g * 0.5
        elif seg1:
            forces[seg1] += g
        elif seg2:
            forces[seg2] += g
    for key in forces:
        forces[key] *= CONTACT_SCALE
    return forces


def get_deflection_forces(data, state):
    forces_g = {}
    for f in FINGERS:
        free = _get_free_angles(state['closure_pct'], MAX_MOTOR_DEG)
        fi   = FINGERS.index(f)
        for j in range(4):
            idx   = fi * 4 + j
            delta = abs(free[idx] - data.qpos[_jnt_qadr[f'{f}_j{j}']])
            delta = delta if delta > 0.002 else 0.0
            torque = K_JOINTS[j] * delta
            defl_g = torque * DEFLECTION_GAIN
            touch_g = 0.0
            if j > 0:
                tname = f'{f}_touch{j}'
                if tname in _touch_sensor:
                    touch_g = float(data.sensordata[_touch_sensor[tname]]) / 9.81 * 1000.0 * CONTACT_SCALE
            forces_g[(f, j)] = max(defl_g, touch_g)
    return forces_g


def print_forces(model, data, state):
    sf = get_segment_forces(model, data)
    df = get_deflection_forces(data, state)
    print(f"\n{'─'*60}")
    print(f"  SEGMENT FORCES @ t={data.time:.2f}s  closure={state['closure_pct']:.1f}%")
    print(f"{'─'*60}")
    for f in FINGERS:
        parts = []
        for j in range(4):
            g = max(sf.get((f, j), 0), df.get((f, j), 0))
            if g > 0.1:
                parts.append(f"L{j}={g:.1f}g")
        print(f"  Finger {f}: {', '.join(parts) if parts else 'no contact'}")
    print(f"{'─'*60}\n")


 
#  IK SOLVER
 
def get_tcp_pose(data):
    return data.xpos[_tcp_body_id].copy(), data.xmat[_tcp_body_id].reshape(3, 3).copy()


def _rot_err_vec(R_target, R_current):
    """Axis-angle orientation error (world frame) from current → target rotation."""
    R_err  = R_target @ R_current.T
    cos_a  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle  = np.arccos(cos_a)
    if abs(angle) < 1e-7:
        return np.zeros(3)
    s = 2.0 * np.sin(angle)
    return (angle / s) * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def ik_step_online(model, data, target_pos, arm_qpos_target,
                   gain=IK_POS_GAIN, target_rot=None, ori_gain=4.0):
    """
    Damped-least-squares IK with Z-clamping and improved convergence.
    Provide target_rot (3×3 ndarray, columns = body axes in world frame)
    to enable 6-DOF position+orientation control; otherwise 3-DOF position only.
    """
    #    Clamp target Z above table   
    clamped_pos = target_pos.copy()
    clamped_pos[2] = max(clamped_pos[2], TCP_Z_MIN)

    tcp_pos = data.xpos[_tcp_body_id]
    tcp_mat = data.xmat[_tcp_body_id].reshape(3, 3)
    dx      = clamped_pos - tcp_pos
    pos_err = np.linalg.norm(dx)

    dof_ids = np.array(_arm_dof_ids)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, _tcp_body_id)
    Jp = jacp[:, dof_ids]

    if target_rot is not None:
        Jr  = jacr[:, dof_ids]
        J   = np.vstack([Jp, Jr])
        dw  = _rot_err_vec(target_rot, tcp_mat)
        ori_err = np.linalg.norm(dw)
        # Adaptive orientation gain: boost when far from target, ease off when close
        eff_ori_gain = ori_gain * min(1.0, ori_err / 0.3 + 0.5)
        v   = np.concatenate([gain * dx, eff_ori_gain * dw])
        # Adaptive damping: more damping near singularities
        lam = IK_LAMBDA * (1.0 + 2.0 * max(0, 1.0 - pos_err / 0.05))
        JJT = J @ J.T + lam * np.eye(6)
        err = np.sqrt(pos_err**2 + ori_err**2)
    else:
        J   = Jp
        v   = gain * dx
        JJT = J @ J.T + IK_LAMBDA * np.eye(3)
        err = pos_err

    if err < 0.0003:
        return err

    dq      = J.T @ np.linalg.solve(JJT, v)
    dq_norm = np.linalg.norm(dq)
    max_step = 2.5
    if dq_norm > max_step:
        dq *= max_step / dq_norm
    dt = model.opt.timestep
    for i in range(N_ARM):
        arm_qpos_target[i] += dq[i] * dt
    return err


def tcp_at_target(data, target_pos, tol=0.012):
    return np.linalg.norm(data.xpos[_tcp_body_id] - target_pos) < tol


def orientation_error(data, target_rot):
    """Return orientation error in radians (0 = perfect alignment)."""
    if target_rot is None:
        return 0.0
    tcp_mat = data.xmat[_tcp_body_id].reshape(3, 3)
    err_vec = _rot_err_vec(target_rot, tcp_mat)
    return np.linalg.norm(err_vec)


def tcp_at_target_with_orientation(data, target_pos, target_rot, pos_tol=0.012, ori_tol=0.15):
    """Check if TCP is at target position AND orientation.
    ori_tol is in radians (~8.6 degrees for 0.15 rad).
    """
    pos_ok = np.linalg.norm(data.xpos[_tcp_body_id] - target_pos) < pos_tol
    ori_ok = orientation_error(data, target_rot) < ori_tol
    return pos_ok and ori_ok


 
#  OBJECT HELPERS
 
def park_all_objects(data):
    """Park inactive objects under the table out of the way."""
    for i, oname in enumerate(OBJ_NAMES):
        qa = _obj_jnt_adr[oname]
        data.qpos[qa:qa+3]   = [5.0 + i*0.2, 5.0, -2.0]
        data.qpos[qa+3:qa+7] = OBJ_QUAT[oname]
        data.qvel[_obj_jnt_vadr[oname]:_obj_jnt_vadr[oname]+6] = 0.0


def place_object_in_pick_tray(data, obj_name):
    """Reset the selected object into the pick tray."""
    qa = _obj_jnt_adr[obj_name]
    pos = obj_pick_pos(obj_name)
    data.qpos[qa:qa+3]   = pos
    data.qpos[qa+3:qa+7] = OBJ_QUAT[obj_name]
    data.qvel[_obj_jnt_vadr[obj_name]:_obj_jnt_vadr[obj_name]+6] = 0.0


 
#  PICK & PLACE STATE MACHINE
 
class PickPlaceSM:
    """
    Drives the arm through the pick-and-place waypoint sequence.
    Thread-safe via a simple lock (reads/writes go through state dict).
    """

    REACH_TOL    = 0.020   # metres — "arrived at waypoint"
    GRASP_STEPS  = 180     # sim steps to ramp close (at grasp step, ~0.18s)
    RELEASE_STEPS= 120

    def __init__(self, model, data, state, arm_cart_target, arm_q_target, home_pos):
        self.model           = model
        self.data            = data
        self.state           = state
        self.arm_cart_target = arm_cart_target
        self.arm_q_target    = arm_q_target
        self.home_pos        = home_pos.copy()

        self._phase_idx   = 0          # index into PHASES
        self._phase_steps = 0          # steps spent in current phase
        self._running     = False      # auto mode
        self._step_request= False      # step mode: advance one phase
        self._lock        = threading.Lock()

        self.obj_name     = OBJ_NAMES[0]
        self._waypoints   = {}
        self._rebuild_waypoints()

    #    public API (called from tkinter thread)               
    def set_object(self, obj_name):
        with self._lock:
            self.obj_name = obj_name
            self._rebuild_waypoints()

    def start_auto(self):
        with self._lock:
            self._phase_idx    = 0
            self._phase_steps  = 0
            self._running      = True
            self._step_request = False
            self._rebuild_waypoints()
            # DON'T call _apply_phase here — it writes arm_cart_target
            # which races with the MuJoCo thread.  tick() will call it
            # on the first frame when it sees _phase_steps == 0.

    def stop(self):
        with self._lock:
            self._running      = False
            self._step_request = False

    def request_step(self):
        """Advance one phase (step-through mode)."""
        with self._lock:
            if not self._running:
                self._step_request = True

    def reset(self):
        with self._lock:
            self._running      = False
            self._step_request = False
            self._phase_idx    = 0
            self._phase_steps  = 0

    def current_phase(self):
        with self._lock:
            return PHASES[self._phase_idx]

    def is_running(self):
        with self._lock:
            return self._running

    #    internal                                             ─
    def _rebuild_waypoints(self):
        pick  = obj_pick_pos(self.obj_name)      # XYZ centre of object on pedestal
        place = obj_place_pos(self.obj_name)      # XYZ drop point above place box
        grasp_z = max(pick[2], TCP_Z_MIN)         # grasp at object's centre height
        drop_z  = place[2]                        # drop height (above box walls)
        hover_z = TABLE_Z + SIDE_HOVER_HEIGHT     # safe hover above table
        transit_z = TABLE_Z + TRANSIT_HEIGHT      # clearance for transit

        if self.obj_name in SIDE_APPROACH_OBJS:
            #    Side approach: pick from open pedestal                 ─
            #   1. HOME      → home (Down orientation)
            #   2. PRE_PICK  → HIGH above & behind object
            #   3. PICK      → DESCEND to grasp height, still behind
            #   4. GRASP     → APPROACH forward to object + CLOSE fingers
            #   5. LIFT      → straight up
            #   6. TRANSIT   → high above & behind place box (from -Y)
            #   7. PRE_PLACE → forward+down to drop point above box centre
            #   8. PLACE     → open fingers → object drops into box
            #   9. RELEASE   → retreat behind box then rise
            #  10. RETURN    → home

            grasp_y = pick[1] - TCP_APPROACH_OFFSET

            self._waypoints = {
                'HOME':      self.home_pos.copy(),
                'PRE_PICK':  np.array([pick[0],
                                       pick[1] - SIDE_APPROACH_OFFSET,
                                       hover_z]),
                'PICK':      np.array([pick[0],
                                       pick[1] - SIDE_APPROACH_OFFSET,
                                       grasp_z]),
                'GRASP':     np.array([pick[0], grasp_y, grasp_z]),
                'LIFT':      np.array([pick[0], grasp_y, transit_z]),
                # Arrive high above & behind the place box
                'TRANSIT':   np.array([place[0],
                                       place[1] - SIDE_APPROACH_OFFSET,
                                       transit_z]),
                # Move forward + descend to drop point above box centre
                # (arm approaches diagonally from behind, clearing the walls)
                'PRE_PLACE': np.array([place[0],
                                       place[1] - TCP_APPROACH_OFFSET,
                                       drop_z]),
                'PLACE':     None,   # open fingers — object drops into box
                # Retreat behind box then rise to transit height
                'RELEASE':   np.array([place[0],
                                       place[1] - SIDE_RETREAT_OFFSET,
                                       transit_z]),
                'RETURN':    self.home_pos.copy(),
            }
        else:
            #    Top-down approach (sphere / compact objects)           ─
            self._waypoints = {
                'HOME':      self.home_pos.copy(),
                'PRE_PICK':  np.array([pick[0],  pick[1],  TABLE_Z + GRASP_HOVER]),
                'PICK':      np.array([pick[0],  pick[1],  max(TABLE_Z + GRASP_DESCEND, TCP_Z_MIN)]),
                'GRASP':     None,   # close fingers only
                'LIFT':      np.array([pick[0],  pick[1],  transit_z]),
                'TRANSIT':   np.array([place[0], place[1], transit_z]),
                'PRE_PLACE': np.array([place[0], place[1], drop_z]),
                'PLACE':     None,   # open fingers — drop into box
                'RELEASE':   np.array([place[0], place[1], transit_z]),
                'RETURN':    self.home_pos.copy(),
            }

    def _apply_phase(self):
        """Set arm target, gripper state, and orientation for current phase.
        Called ONLY from the MuJoCo thread (inside tick → inside v.lock).
        """
        phase = PHASES[self._phase_idx]
        wp    = self._waypoints.get(phase)
        if wp is not None:
            self.arm_cart_target[:] = wp
            self.state['tracking_mocap'] = False

        #    Command orientation target                               ─
        # The IK tracks this continuously every frame.  We never WAIT
        # for convergence — we just set the target and let the arm
        # converge while it travels to the next waypoint.
        if self._running or self._step_request:
            best_ori = OBJ_DEFAULT_ORI.get(self.obj_name, 'Down')
            if phase in ('HOME', 'RETURN'):
                self.state['tcp_rot_target'] = ROT_GRIPPER_DOWN
            else:
                self.state['tcp_rot_target'] = ORI_MATRICES.get(best_ori, ROT_GRIPPER_DOWN)

        #    Gripper state                                             
        if phase == 'PLACE':
            self.state['releasing'] = True
            self.state['grasping']  = False
        if phase in ('HOME', 'PRE_PICK'):
            self.state['closure_pct'] = 0.0
            self.state['grasping']    = False
            self.state['releasing']   = False

        print(f"  >> Phase: {phase}")

    def _phase_done(self):
        """Return True when the current phase can advance.

        IMPORTANT: We NEVER block on orientation convergence.
        The IK continuously tracks the orientation target every frame.
        Phase advancement is gated ONLY on:
          - Position reaching the waypoint (for movement phases)
          - Gripper fully closed (GRASP) or fully opened (PLACE)
        """
        phase = PHASES[self._phase_idx]
        wp    = self._waypoints.get(phase)

        #    GRASP: approach to waypoint THEN close fingers         
        if phase == 'GRASP':
            if wp is not None:
                if not tcp_at_target(self.data, wp, tol=0.025):
                    return False   # still approaching

            # Arm arrived (or no waypoint) — trigger finger closing
            if not self.state['grasping'] and self.state['closure_pct'] < 100.0:
                self.state['grasping']  = True
                self.state['releasing'] = False
                print("  >> Fingers closing…")

            return self.state['closure_pct'] >= 100.0 and not self.state['grasping']

        #    PLACE (drop): wait for full finger opening             
        if phase == 'PLACE':
            return self.state['closure_pct'] <= 0.0 and not self.state['releasing']

        #    All movement phases: position only                     
        if wp is not None:
            return tcp_at_target(self.data, wp, tol=self.REACH_TOL)
        return True  # instant phases (None waypoint)

    def tick(self):
        """
        Called once per simulation frame (inside viewer lock).
        Advances the state machine in auto mode, or on step request.
        """
        with self._lock:
            if not self._running and not self._step_request:
                return

            #    First step: apply the current phase if not yet applied   
            if self._phase_steps == 0:
                self._apply_phase()

            self._phase_steps += 1

            if self._phase_done():
                # Advance to next phase
                next_idx = self._phase_idx + 1
                if next_idx >= len(PHASES):
                    # Sequence complete
                    self._running      = False
                    self._step_request = False
                    self._phase_idx    = 0
                    self._phase_steps  = 0
                    print("\n  ✅ Pick & Place sequence complete!\n")
                    return
                self._phase_idx   = next_idx
                self._phase_steps = 0
                self._apply_phase()

                if not self._running:
                    # Step mode: consume the request and pause
                    self._step_request = False


 
#  TKINTER CONTROL PANEL
 
class ControlPanel:
    def __init__(self, sm: PickPlaceSM, state: dict):
        self.sm    = sm
        self.state = state

        self.root = tk.Tk()
        self.root.title("Pick & Place Control")
        self.root.resizable(False, False)
        self.root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='#2e2e3e',
                        background='#2e2e3e', foreground='#e0e0f0',
                        selectbackground='#4a4a6a')

        pad = dict(padx=10, pady=6)

        #    Title                                           
        tk.Label(self.root, text="UR5e Pick & Place",
                 font=('Helvetica', 14, 'bold'),
                 fg='#c0d8f8', bg='#1e1e2e').pack(**pad)

        #    Object selector                                 
        frame_obj = tk.Frame(self.root, bg='#1e1e2e')
        frame_obj.pack(fill='x', **pad)
        tk.Label(frame_obj, text="Object:", width=9, anchor='w',
                 fg='#a0b8d8', bg='#1e1e2e').pack(side='left')
        self.obj_var = tk.StringVar(value=OBJ_DISPLAY[OBJ_NAMES[0]])
        self.obj_combo = ttk.Combobox(
            frame_obj, textvariable=self.obj_var,
            values=[OBJ_DISPLAY[n] for n in OBJ_NAMES],
            state='readonly', width=16)
        self.obj_combo.pack(side='left', padx=(4, 0))
        self.obj_combo.bind('<<ComboboxSelected>>', self._on_obj_change)

        #    Orientation selector                             
        frame_ori = tk.Frame(self.root, bg='#1e1e2e')
        frame_ori.pack(fill='x', **pad)
        tk.Label(frame_ori, text="Orientation:", width=9, anchor='w',
                 fg='#a0b8d8', bg='#1e1e2e').pack(side='left')
        self.ori_var = tk.StringVar(value="Side-Perp")
        self.ori_combo = ttk.Combobox(
            frame_ori, textvariable=self.ori_var,
            values=["Down", "Side", "Side-Perp"],
            state='readonly', width=16)
        self.ori_combo.pack(side='left', padx=(4, 0))
        self.ori_combo.bind('<<ComboboxSelected>>', self._on_ori_change)

        #    Phase indicator                                 
        tk.Label(self.root, text="Current Phase",
                 fg='#a0b8d8', bg='#1e1e2e',
                 font=('Helvetica', 9)).pack()
        self.phase_label = tk.Label(
            self.root, text="🏠 Home",
            font=('Helvetica', 13, 'bold'),
            fg='#ffffff', bg='#4a90d9',
            width=22, pady=6, relief='flat')
        self.phase_label.pack(padx=10, pady=4)

        # Phase progress bar (simple coloured boxes)
        self.phase_frame = tk.Frame(self.root, bg='#1e1e2e')
        self.phase_frame.pack(fill='x', padx=10, pady=4)
        self.phase_boxes = []
        cols = 5
        for i, ph in enumerate(PHASES):
            row, col = divmod(i, cols)
            lbl = tk.Label(
                self.phase_frame,
                text=PHASE_LABELS[ph].split(' ', 1)[1] if ' ' in PHASE_LABELS[ph] else ph,
                font=('Helvetica', 7),
                fg='#ffffff', bg='#3a3a4a',
                width=9, pady=2, relief='flat')
            lbl.grid(row=row, column=col, padx=2, pady=2)
            self.phase_boxes.append((ph, lbl))

        #    Control buttons                               ─
        btn_cfg = dict(font=('Helvetica', 10, 'bold'),
                       width=12, relief='flat', cursor='hand2',
                       padx=4, pady=6)

        frame_btns = tk.Frame(self.root, bg='#1e1e2e')
        frame_btns.pack(fill='x', padx=10, pady=6)

        self.btn_auto = tk.Button(
            frame_btns, text="▶  Auto Run",
            bg='#2e7d32', fg='white',
            activebackground='#388e3c',
            command=self._on_auto, **btn_cfg)
        self.btn_auto.grid(row=0, column=0, padx=4, pady=3)

        self.btn_step = tk.Button(
            frame_btns, text="⏭  Step",
            bg='#1565c0', fg='white',
            activebackground='#1976d2',
            command=self._on_step, **btn_cfg)
        self.btn_step.grid(row=0, column=1, padx=4, pady=3)

        self.btn_stop = tk.Button(
            frame_btns, text="⏹  Stop",
            bg='#b71c1c', fg='white',
            activebackground='#c62828',
            command=self._on_stop, **btn_cfg)
        self.btn_stop.grid(row=1, column=0, padx=4, pady=3)

        self.btn_reset = tk.Button(
            frame_btns, text="↺  Reset",
            bg='#4a4a5a', fg='white',
            activebackground='#5a5a6a',
            command=self._on_reset, **btn_cfg)
        self.btn_reset.grid(row=1, column=1, padx=4, pady=3)

        #    Status                                         ─
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 fg='#90aac8', bg='#1e1e2e',
                 font=('Helvetica', 9, 'italic'),
                 wraplength=240).pack(padx=10, pady=(4, 2))

        #    Gripper slider                                 ─
        frame_grip = tk.Frame(self.root, bg='#1e1e2e')
        frame_grip.pack(fill='x', padx=10, pady=(2, 8))
        tk.Label(frame_grip, text="Closure %",
                 fg='#a0b8d8', bg='#1e1e2e', width=10, anchor='w').pack(side='left')
        self.grip_bar = ttk.Progressbar(frame_grip, length=140,
                                        mode='determinate', maximum=100)
        self.grip_bar.pack(side='left', padx=4)
        self.grip_pct_lbl = tk.Label(frame_grip, text="0%",
                                     fg='#e0e0f0', bg='#1e1e2e', width=5)
        self.grip_pct_lbl.pack(side='left')

        # Keyboard hint
        tk.Label(self.root, text="Keyboard: WASD/RF=move  Q=grasp  U=release\n"
                                 "T=mocap  /=forces  Backspace=reset",
                 fg='#505070', bg='#1e1e2e',
                 font=('Helvetica', 8)).pack(pady=(0, 6))

        self._update()

    #    Callbacks                                         ─
    def _on_obj_change(self, _evt=None):
        try:
            label = self.obj_var.get()
            name  = next(k for k, v in OBJ_DISPLAY.items() if v == label)
            self.sm.set_object(name)
            best_ori = OBJ_DEFAULT_ORI.get(name, 'Down')
            self.ori_var.set(best_ori)
            self.state['tcp_rot_target'] = ORI_MATRICES[best_ori]
            self.state['_obj_change_requested'] = name
            self.status_var.set(f"Object: {label}")
            print(f"  >> Object: {label}, auto-orientation: {best_ori}")
        except Exception as e:
            print(f"  !! obj_change error: {e}")

    def _on_ori_change(self, _evt=None):
        try:
            ori = self.ori_var.get()
            mat = ORI_MATRICES.get(ori, ROT_GRIPPER_DOWN)
            self.state['tcp_rot_target'] = mat
            self.status_var.set(f"Orientation: {ori}")
        except Exception as e:
            print(f"  !! ori_change error: {e}")

    def _on_auto(self):
        try:
            self.sm.start_auto()
            self.status_var.set("Auto run started…")
            print("  >> AUTO RUN started")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            print(f"  !! Auto start error: {e}")

    def _on_step(self):
        try:
            self.sm.request_step()
            self.status_var.set("Step requested")
            print("  >> STEP requested")
        except Exception as e:
            print(f"  !! Step error: {e}")

    def _on_stop(self):
        try:
            self.sm.stop()
            self.status_var.set("Stopped")
            print("  >> STOPPED")
        except Exception as e:
            print(f"  !! Stop error: {e}")

    def _on_reset(self):
        try:
            self.sm.reset()
            self.state['closure_pct'] = 0.0
            self.state['grasping']    = False
            self.state['releasing']   = False
            self.state['_reset_requested'] = True
            self.status_var.set("Reset — press Auto or Step")
            print("  >> RESET requested")
        except Exception as e:
            print(f"  !! Reset error: {e}")

    #    Periodic UI refresh                               ─
    def _update(self):
        try:
            phase = self.sm.current_phase()
            label = PHASE_LABELS.get(phase, phase)
            color = PHASE_COLORS.get(phase, '#4a4a5a')
            self.phase_label.config(text=label, bg=color)

            for ph, lbl in self.phase_boxes:
                if ph == phase:
                    lbl.config(bg=PHASE_COLORS.get(ph, '#4a4a5a'), fg='#ffffff')
                else:
                    lbl.config(bg='#2a2a3a', fg='#606080')

            pct = self.state.get('closure_pct', 0.0)
            self.grip_bar['value'] = pct
            self.grip_pct_lbl.config(text=f"{pct:.0f}%")

            if self.sm.is_running():
                self.status_var.set(f"Running: {label}")
        except Exception:
            pass  # UI update failure is non-critical

        self.root.after(80, self._update)

    def run(self):
        self.root.mainloop()


 
#  MAIN
 
HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║          UR5e + 3-Finger Gripper — Pick & Place         ║
╠══════════════════════════════════════════════════════════╣
║  ARM CARTESIAN                                           ║
║    W/S    — TCP +Y / -Y  (forward/back)                 ║
║    A/D    — TCP -X / +X  (left/right)                   ║
║    R/F    — TCP +Z / -Z  (up/down)                      ║
║                                                          ║
║  GRIPPER                                                 ║
║    Q      — Grasp (ramp close)                          ║
║    U      — Release (ramp open)                         ║
║    /      — Print segment forces                        ║
║    +/-    — Max closure ±50°                            ║
║                                                          ║
║  TARGET TRACKING                                         ║
║    T      — Toggle mocap drag (green marker)            ║
║                                                          ║
║  AUTOMATION (use the Control Panel)                     ║
║    Auto Run  — full pick & place sequence               ║
║    Step      — advance one phase at a time              ║
║    Stop/Reset — abort or restart                        ║
║                                                          ║
║  GENERAL                                                 ║
║    H          — Print this help                         ║
║    Backspace  — Reset simulation                        ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, 'ur5e_gripper.xml')
    if not os.path.exists(xml_path):
        print(f"ERROR: Cannot find {xml_path}")
        sys.exit(1)

    print("Loading model …")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    _build_lookups(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    arm_q_target    = np.array([data.qpos[qa] for qa in _arm_jnt_qadr])
    arm_cart_target = data.xpos[_tcp_body_id].copy()

    state = {
        'closure_pct':   0.0,
        'grasping':      False,
        'releasing':     False,
        'grasp_speed':   0.4,
        'splay_deg':     0.0,
        'max_motor_deg': MAX_MOTOR_DEG,
        'tracking_mocap': False,
        # Default to Side-Perp orientation (perpendicular side grasp)
        'tcp_rot_target': ROT_GRIPPER_SIDE_Y_PERP,
        # Thread-safe reset/obj-change flags
        '_reset_requested': False,
        '_obj_change_requested': None,
    }

    calibrate_free_angles(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    arm_q_target[:]    = [data.qpos[qa] for qa in _arm_jnt_qadr]
    arm_cart_target[:] = data.xpos[_tcp_body_id]

    # Park non-selected objects; selected one goes into pick tray
    park_all_objects(data)
    place_object_in_pick_tray(data, OBJ_NAMES[0])
    mujoco.mj_forward(model, data)

    home_pos = data.xpos[_tcp_body_id].copy()

    #    State machine & mocap                               ─
    sm       = PickPlaceSM(model, data, state, arm_cart_target, arm_q_target, home_pos)
    mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_marker')]
    data.mocap_pos[mocap_id] = arm_cart_target.copy()

    #    Helper: perform sim reset (called from MuJoCo thread only)   
    def _do_sim_reset(sel_name=None):
        """Reset simulation state. Must be called from MuJoCo thread."""
        mujoco.mj_resetDataKeyframe(model, data, 0)
        park_all_objects(data)
        if sel_name:
            place_object_in_pick_tray(data, sel_name)
        else:
            place_object_in_pick_tray(data, OBJ_NAMES[0])
        state['closure_pct'] = 0.0
        state['grasping'] = state['releasing'] = False
        state['tracking_mocap'] = False
        arm_q_target[:]    = [data.qpos[qa] for qa in _arm_jnt_qadr]
        arm_cart_target[:] = data.xpos[_tcp_body_id]
        mujoco.mj_forward(model, data)

    #    Key callback                                       ─
    def key_cb(key):
        k = chr(key) if 32 <= key < 127 else key
        step = 0.003
        if   k in ('W','w'): arm_cart_target[1] += step; state['tracking_mocap'] = False
        elif k in ('S','s'): arm_cart_target[1] -= step; state['tracking_mocap'] = False
        elif k in ('A','a'): arm_cart_target[0] -= step; state['tracking_mocap'] = False
        elif k in ('D','d'): arm_cart_target[0] += step; state['tracking_mocap'] = False
        elif k in ('R','r'): arm_cart_target[2] += step; state['tracking_mocap'] = False
        elif k in ('F','f'):
            arm_cart_target[2] = max(arm_cart_target[2] - step, TCP_Z_MIN)
            state['tracking_mocap'] = False
        elif k in ('Q','q'):
            state['grasping']  = True;  state['releasing'] = False
            print("  >> GRASPING")
        elif k in ('U','u'):
            state['releasing'] = True;  state['grasping']  = False
            print("  >> RELEASING")
        elif k in ('T','t'):
            state['tracking_mocap'] = not state['tracking_mocap']
            if state['tracking_mocap']:
                data.mocap_pos[mocap_id] = data.xpos[_tcp_body_id].copy()
                print("  >> Mocap tracking ON")
            else:
                arm_cart_target[:] = data.xpos[_tcp_body_id].copy()
                print("  >> Mocap tracking OFF")
        elif k in ('/', '?'):
            print_forces(model, data, state)
        elif k in ('H','h'):
            print(HELP_TEXT)
        elif k in ('+','='):
            state['max_motor_deg'] = min(state['max_motor_deg'] + 50, 700)
            print(f"  >> Max closure: {state['max_motor_deg']:.0f}°")
        elif k in ('-','_'):
            state['max_motor_deg'] = max(state['max_motor_deg'] - 50, 450)
            print(f"  >> Max closure: {state['max_motor_deg']:.0f}°")
        elif key == 65288:  # Backspace
            sm.reset()
            state['_reset_requested'] = True
            print("  >> RESET requested")

    #    Gripper & arm apply                                 
    def apply_gripper():
        mmd   = state['max_motor_deg']
        dphi  = np.deg2rad(state['closure_pct'] / 100.0 * mmd)
        target= EFFECTIVE_RP * dphi
        splay = np.deg2rad(state['splay_deg'])
        for f in FINGERS:
            data.ctrl[_act_pos[f]] = target
            data.ctrl[_act_frc[f]] = 0.0
            if f in _act_splay:
                data.ctrl[_act_splay[f]] = splay

    def apply_arm():
        for i, aid in enumerate(_arm_act_ids):
            data.ctrl[aid] = arm_q_target[i]

    #    Print startup info                                 
    print("\n" + "="*60)
    print("  UR5e + 3-FINGER GRIPPER — Pick & Place")
    print("="*60)
    tcp = data.xpos[_tcp_body_id]
    print(f"  TCP home: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]")
    print(f"  Pick tray:  x={PICK_TRAY_XY[0]:.2f}, y={PICK_TRAY_XY[1]:.2f}")
    print(f"  Place tray: x={PLACE_TRAY_XY[0]:.2f}, y={PLACE_TRAY_XY[1]:.2f}")
    print(f"  Press H for keyboard controls")
    print("="*60 + "\n")

    #    Control panel (must be created on main thread on Windows)   
    panel = ControlPanel(sm, state)

    # Wire up object-change: set a flag for the MuJoCo thread to handle safely
    _orig_obj_change = panel._on_obj_change
    def _patched_obj_change(evt=None):
        _orig_obj_change(evt)
        label = panel.obj_var.get()
        sel   = next(k for k, v in OBJ_DISPLAY.items() if v == label)
        state['_obj_change_requested'] = sel
    panel._on_obj_change = _patched_obj_change
    panel.obj_combo.bind('<<ComboboxSelected>>', _patched_obj_change)

    #    MuJoCo loop runs in a background thread             
    # (Tkinter mainloop must own the main thread on Windows)
    import mujoco.viewer as viewer
    n_substeps = 10

    def _mujoco_loop():
        with viewer.launch_passive(model, data, key_callback=key_cb) as v:
            v.cam.lookat[:] = [0.15, 0.20, 0.52]
            v.cam.distance  = 1.85
            v.cam.elevation = -25
            v.cam.azimuth   = 155

            last_print = 0.0

            while v.is_running():
                t0 = time.time()

                with v.lock():
                    #    Handle reset request (from UI or keyboard)   
                    if state.get('_reset_requested'):
                        state['_reset_requested'] = False
                        try:
                            sel = next(k for k, v_ in OBJ_DISPLAY.items()
                                       if v_ == panel.obj_var.get())
                        except Exception:
                            sel = OBJ_NAMES[0]
                        _do_sim_reset(sel)
                        print("  >> RESET done")

                    #    Handle object change request               
                    obj_req = state.get('_obj_change_requested')
                    if obj_req is not None:
                        state['_obj_change_requested'] = None
                        park_all_objects(data)
                        place_object_in_pick_tray(data, obj_req)
                        mujoco.mj_forward(model, data)

                    #    Gripper ramp                           
                    if state['grasping']:
                        state['closure_pct'] = min(state['closure_pct'] + state['grasp_speed'], 100.0)
                        if state['closure_pct'] >= 100.0:
                            state['grasping'] = False
                            print(f"  >> Grasped at {state['max_motor_deg']:.0f}°")
                            print_forces(model, data, state)
                    elif state['releasing']:
                        state['closure_pct'] = max(state['closure_pct'] - state['grasp_speed'], 0.0)
                        if state['closure_pct'] <= 0.0:
                            state['releasing'] = False
                            print("  >> Released")

                    #    Mocap tracking                         
                    if state['tracking_mocap']:
                        arm_cart_target[:] = data.mocap_pos[mocap_id]
                    else:
                        data.mocap_pos[mocap_id] = arm_cart_target.copy()

                    #    Z safety clamp                         
                    arm_cart_target[2] = max(arm_cart_target[2], TCP_Z_MIN)

                    #    Collision avoidance                     
                    n_coll = check_arm_collisions(model, data, arm_cart_target)

                    #    State machine tick                     
                    sm.tick()

                    #    IK                                     
                    ik_step_online(model, data, arm_cart_target,
                                   arm_q_target, gain=12.0,
                                   target_rot=state.get('tcp_rot_target'),
                                   ori_gain=18.0)

                    #    Apply controls                         
                    apply_gripper()
                    apply_arm()

                    #    Step sim                               
                    for _ in range(n_substeps):
                        mujoco.mj_step(model, data)

                    #    Periodic status print                 ─
                    now = data.time
                    if now - last_print >= 5.0:
                        tcp   = data.xpos[_tcp_body_id]
                        pct   = state['closure_pct']
                        phase = sm.current_phase()
                        ori_err = orientation_error(data, state.get('tcp_rot_target'))
                        ori_deg = np.rad2deg(ori_err)
                        coll_str = f"  coll={n_coll}" if n_coll > 0 else ""
                        print(f"  t={now:.1f}s  TCP=[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]"
                              f"  grip={pct:.0f}%  ori_err={ori_deg:.1f}°  phase={phase}{coll_str}")
                        last_print = now

                v.sync()
                dt = model.opt.timestep * n_substeps - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        print("\nDone.")
        try:
            panel.root.quit()
        except Exception:
            pass

    mujoco_thread = threading.Thread(target=_mujoco_loop, daemon=True)
    mujoco_thread.start()

    #    Tkinter mainloop on main thread (required on Windows)   
    panel.run()


if __name__ == '__main__':
    main()