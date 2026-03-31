"""
Graspable objects for the 3-finger compliant gripper simulation.
"""

import numpy as np

def _cylinder_faces(radius, length, n_seg=36, center=(0, 0, 0),
                    axis='x', color='#666666'):
    """Solid cylinder"""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, n_seg + 1)
    faces = []

    for i in range(n_seg):
        a1, a2 = angles[i], angles[i + 1]
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)

        if axis == 'x':
            p1_lo = [cx - length / 2, cy + radius * c1, cz + radius * s1]
            p2_lo = [cx - length / 2, cy + radius * c2, cz + radius * s2]
            p1_hi = [cx + length / 2, cy + radius * c1, cz + radius * s1]
            p2_hi = [cx + length / 2, cy + radius * c2, cz + radius * s2]
            cap_lo = [cx - length / 2, cy, cz]
            cap_hi = [cx + length / 2, cy, cz]
        elif axis == 'y':
            p1_lo = [cx + radius * c1, cy - length / 2, cz + radius * s1]
            p2_lo = [cx + radius * c2, cy - length / 2, cz + radius * s2]
            p1_hi = [cx + radius * c1, cy + length / 2, cz + radius * s1]
            p2_hi = [cx + radius * c2, cy + length / 2, cz + radius * s2]
            cap_lo = [cx, cy - length / 2, cz]
            cap_hi = [cx, cy + length / 2, cz]
        else:  # 'z'
            p1_lo = [cx + radius * c1, cy + radius * s1, cz - length / 2]
            p2_lo = [cx + radius * c2, cy + radius * s2, cz - length / 2]
            p1_hi = [cx + radius * c1, cy + radius * s1, cz + length / 2]
            p2_hi = [cx + radius * c2, cy + radius * s2, cz + length / 2]
            cap_lo = [cx, cy, cz - length / 2]
            cap_hi = [cx, cy, cz + length / 2]

        faces.append([p1_lo, p2_lo, p2_hi, p1_hi])  # side
        faces.append([cap_hi, p1_hi, p2_hi])          # hi cap
        faces.append([cap_lo, p2_lo, p1_lo])           # lo cap

    return faces, [color] * len(faces)


def _frustum_faces(r_lo, r_hi, length, n_seg=36, center=(0, 0, 0),
                   axis='x', color='#666666'):
    """Tapered cylinder
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, n_seg + 1)
    faces = []

    for i in range(n_seg):
        a1, a2 = angles[i], angles[i + 1]
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)

        if axis == 'x':
            p1_lo = [cx - length / 2, cy + r_lo * c1, cz + r_lo * s1]
            p2_lo = [cx - length / 2, cy + r_lo * c2, cz + r_lo * s2]
            p1_hi = [cx + length / 2, cy + r_hi * c1, cz + r_hi * s1]
            p2_hi = [cx + length / 2, cy + r_hi * c2, cz + r_hi * s2]
            cap_lo = [cx - length / 2, cy, cz]
            cap_hi = [cx + length / 2, cy, cz]
        elif axis == 'y':
            p1_lo = [cx + r_lo * c1, cy - length / 2, cz + r_lo * s1]
            p2_lo = [cx + r_lo * c2, cy - length / 2, cz + r_lo * s2]
            p1_hi = [cx + r_hi * c1, cy + length / 2, cz + r_hi * s1]
            p2_hi = [cx + r_hi * c2, cy + length / 2, cz + r_hi * s2]
            cap_lo = [cx, cy - length / 2, cz]
            cap_hi = [cx, cy + length / 2, cz]
        else:  # 'z'
            p1_lo = [cx + r_lo * c1, cy + r_lo * s1, cz - length / 2]
            p2_lo = [cx + r_lo * c2, cy + r_lo * s2, cz - length / 2]
            p1_hi = [cx + r_hi * c1, cy + r_hi * s1, cz + length / 2]
            p2_hi = [cx + r_hi * c2, cy + r_hi * s2, cz + length / 2]
            cap_lo = [cx, cy, cz - length / 2]
            cap_hi = [cx, cy, cz + length / 2]

        faces.append([p1_lo, p2_lo, p2_hi, p1_hi])  # side
        faces.append([cap_hi, p1_hi, p2_hi])          # hi cap
        faces.append([cap_lo, p2_lo, p1_lo])           # lo cap

    return faces, [color] * len(faces)


def _annulus_faces(outer_r, inner_r, thickness, n_seg=48,
                   center=(0, 0, 0), axis='z', color='#c8a84a'):
    """Annular ring"""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, n_seg + 1)
    faces = []
    ht = thickness / 2

    for i in range(n_seg):
        a1, a2 = angles[i], angles[i + 1]
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)

        if axis == 'z':
            faces.append([  # outer wall
                [cx + outer_r * c1, cy + outer_r * s1, cz - ht],
                [cx + outer_r * c2, cy + outer_r * s2, cz - ht],
                [cx + outer_r * c2, cy + outer_r * s2, cz + ht],
                [cx + outer_r * c1, cy + outer_r * s1, cz + ht]])
            faces.append([  # inner wall
                [cx + inner_r * c1, cy + inner_r * s1, cz - ht],
                [cx + inner_r * c1, cy + inner_r * s1, cz + ht],
                [cx + inner_r * c2, cy + inner_r * s2, cz + ht],
                [cx + inner_r * c2, cy + inner_r * s2, cz - ht]])
            faces.append([  # top
                [cx + outer_r * c1, cy + outer_r * s1, cz + ht],
                [cx + outer_r * c2, cy + outer_r * s2, cz + ht],
                [cx + inner_r * c2, cy + inner_r * s2, cz + ht],
                [cx + inner_r * c1, cy + inner_r * s1, cz + ht]])
            faces.append([  # bottom
                [cx + outer_r * c1, cy + outer_r * s1, cz - ht],
                [cx + inner_r * c1, cy + inner_r * s1, cz - ht],
                [cx + inner_r * c2, cy + inner_r * s2, cz - ht],
                [cx + outer_r * c2, cy + outer_r * s2, cz - ht]])

        elif axis == 'x':
            faces.append([
                [cx - ht, cy + outer_r * c1, cz + outer_r * s1],
                [cx - ht, cy + outer_r * c2, cz + outer_r * s2],
                [cx + ht, cy + outer_r * c2, cz + outer_r * s2],
                [cx + ht, cy + outer_r * c1, cz + outer_r * s1]])
            faces.append([
                [cx - ht, cy + inner_r * c1, cz + inner_r * s1],
                [cx + ht, cy + inner_r * c1, cz + inner_r * s1],
                [cx + ht, cy + inner_r * c2, cz + inner_r * s2],
                [cx - ht, cy + inner_r * c2, cz + inner_r * s2]])
            faces.append([
                [cx + ht, cy + outer_r * c1, cz + outer_r * s1],
                [cx + ht, cy + outer_r * c2, cz + outer_r * s2],
                [cx + ht, cy + inner_r * c2, cz + inner_r * s2],
                [cx + ht, cy + inner_r * c1, cz + inner_r * s1]])
            faces.append([
                [cx - ht, cy + outer_r * c1, cz + outer_r * s1],
                [cx - ht, cy + inner_r * c1, cz + inner_r * s1],
                [cx - ht, cy + inner_r * c2, cz + inner_r * s2],
                [cx - ht, cy + outer_r * c2, cz + outer_r * s2]])

    return faces, [color] * len(faces)



def _build_paint_tube(z_sign=1):
    """acrylic paint
    """
    r_crimp = 0.019      # 19 mm radius at crimped (wide) end
    r_cap   = 0.014      # 14 mm radius near cap (narrow) end
    h_body  = 0.130      # 130 mm tapered body
    # Centre height = radius of widest part -> bottom sits at z~0
    cz = z_sign * r_crimp

    # Tapered tube body  (crimp at -X, cap at +X)
    f1, c1 = _frustum_faces(r_crimp, r_cap, h_body, 32,
                            center=(0, 0, cz), axis='x', color='#1a1a1a')
    # Crimped flat tail
    f2, c2 = _cylinder_faces(r_crimp * 0.90, 0.005, 20,
                             center=(-h_body / 2 - 0.0025, 0, cz),
                             axis='x', color='#333333')
    # Shoulder + nozzle
    f3, c3 = _frustum_faces(r_cap, r_cap * 0.45, 0.014, 24,
                            center=(h_body / 2 + 0.007, 0, cz),
                            axis='x', color='#2a2a2a')
    # Screw cap
    f4, c4 = _cylinder_faces(r_cap * 0.50, 0.012, 20,
                             center=(h_body / 2 + 0.014 + 0.006, 0, cz),
                             axis='x', color='#e0e0e0')
    # Colour label band (teal, matching Liquitex Basics in photo)
    f5, c5 = _frustum_faces(r_crimp * 0.92, r_cap + 0.001, h_body * 0.55, 32,
                            center=(0.008, 0, cz), axis='x',
                            color='#1a8a8a')
    return f1 + f2 + f3 + f4 + f5, c1 + c2 + c3 + c4 + c5


def _build_paper_towel(z_sign=1):
    """Paper towel
    """
    r_outer = 0.0889 / 1.5   # 59.3 mm radius (scaled down)
    r_inner = 0.015           # cardboard tube ~30 mm dia
    h       = 0.3048 / 1.5   # 203 mm length (scaled down)
    cz = z_sign * r_outer    # bottom at z ~ 0

    # Outer paper surface (off-white)
    f1, c1 = _cylinder_faces(r_outer, h, 48,
                             center=(0, 0, cz), axis='x',
                             color='#f5f0e6')
    # Cardboard tube core (visible from ends)
    f2, c2 = _annulus_faces(r_inner + 0.002, r_inner, h + 0.001, 36,
                            center=(0, 0, cz), axis='x',
                            color='#a08050')
    return f1 + f2, c1 + c2


def _build_screwdriver(z_sign=1):
    """Screwdriver
    """
    r_handle = 0.0254 / 2    # 12.7 mm radius  (1 in dia handle)
    h_handle = 0.050          # 50 mm handle length
    r_shaft  = 0.0015         # 1.5 mm shaft radius
    h_shaft  = 0.065          # 65 mm shaft
    cz = z_sign * r_handle    # bottom at z ~ 0

    # Handle (red)
    f1, c1 = _cylinder_faces(r_handle, h_handle, 32,
                             center=(0, 0, cz), axis='x',
                             color='#cc3333')
    # Rubber grip ring
    f2, c2 = _cylinder_faces(r_handle + 0.001, 0.012, 32,
                             center=(-0.008, 0, cz), axis='x',
                             color='#222222')
    # Metal ferrule
    f3, c3 = _cylinder_faces(r_handle * 0.55, 0.006, 24,
                             center=(h_handle / 2 + 0.003, 0, cz),
                             axis='x', color='#aaaaaa')
    # Shaft
    f4, c4 = _cylinder_faces(r_shaft, h_shaft, 16,
                             center=(h_handle / 2 + 0.003 + h_shaft / 2, 0, cz),
                             axis='x', color='#cccccc')
    return f1 + f2 + f3 + f4, c1 + c2 + c3 + c4


def _build_tape_roll(z_sign=1):
    """Tape roll
    """
    outer_r   = 0.047625     # 47.6 mm  (3.75 in / 2)
    inner_r   = 0.0381       # 38.1 mm  (3 in / 2)
    thickness = 0.0508       # 50.8 mm  (2 in)

    cz = z_sign * (thickness / 2)

    f1, c1 = _annulus_faces(outer_r, inner_r, thickness, 48,
                            center=(0, 0, cz), axis='z',
                            color='#c8a84a')

    f2, c2 = _annulus_faces(inner_r, inner_r - 0.002, thickness + 0.001, 36,
                            center=(0, 0, cz), axis='z',
                            color='#8a7040')
    return f1 + f2, c1 + c2



_BUILDERS = {
    'None':        None,
    'Paint Tube':  _build_paint_tube,
    'Paper Towel': _build_paper_towel,
    'Screwdriver': _build_screwdriver,
    'Tape Roll':   _build_tape_roll,
}

OBJECT_NAMES = list(_BUILDERS.keys())


def build_object(name, z_sign=1, offset=(0.0, 0.0, 0.0), scale=1.0,
                 rotation_deg=(0.0, 0.0, 0.0)):
   
    builder = _BUILDERS.get(name)
    if builder is None:
        return None, None
    faces, fcolors = builder(z_sign=z_sign)

    rx_d, ry_d, rz_d = rotation_deg
    apply_rot = any(abs(a) > 1e-9 for a in (rx_d, ry_d, rz_d))

    if apply_rot:
        rx = np.deg2rad(rx_d)
        ry = np.deg2rad(ry_d)
        rz = np.deg2rad(rz_d)
        Rx = np.array([[1,           0,            0],
                       [0,  np.cos(rx), -np.sin(rx)],
                       [0,  np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0,           1,           0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0,           0,           1]])
        R = Rz @ Ry @ Rx

        # Rotate about the mesh geometric centroid so the object stays centred
        all_pts = np.array([pt for face in faces for pt in face])
        centroid = all_pts.mean(axis=0)
    else:
        R = None
        centroid = None

    apply_transform = (scale != 1.0
                       or any(o != 0.0 for o in offset)
                       or apply_rot)

    if apply_transform:
        ox, oy, oz = offset
        new_faces = []
        for face in faces:
            new_face = []
            for pt in face:
                p = np.array(pt, dtype=float) * scale
                if apply_rot:
                    p = R @ (p - centroid) + centroid
                new_face.append([p[0] + ox, p[1] + oy, p[2] + oz])
            new_faces.append(new_face)
        faces = new_faces

    return faces, fcolors