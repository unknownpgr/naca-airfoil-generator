import numpy as np


def airfoil(
    x,
    m=0.1,
    p=0.6,
    t=0.15,
    angle_of_attack=0.2,
):
    """
    Define the airfoil shape using the NACA airfoil.
    """

    x = np.array(x, copy=True)

    is_upper = x < 0.5
    x[is_upper] = x[is_upper] * 2  # 0~0.5 -> 0~1
    x[~is_upper] = 2 * (1 - x[~is_upper])  # 0.5~1 -> 1~0

    # Camber line
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    yc[x < p] = m / p**2 * (2 * p * x[x < p] - x[x < p] ** 2)
    yc[x >= p] = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x[x >= p] - x[x >= p] ** 2)
    dyc[x < p] = 2 * m / p**2 * (p - x[x < p])
    dyc[x >= p] = 2 * m / (1 - p) ** 2 * (p - x[x >= p])

    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    # Angle of the camber line
    theta = np.arctan(dyc)

    xp = np.zeros_like(x)
    yp = np.zeros_like(x)
    xp[is_upper] = x[is_upper] - yt[is_upper] * np.sin(theta[is_upper])
    yp[is_upper] = yc[is_upper] + yt[is_upper] * np.cos(theta[is_upper])
    xp[~is_upper] = x[~is_upper] + yt[~is_upper] * np.sin(theta[~is_upper])
    yp[~is_upper] = yc[~is_upper] - yt[~is_upper] * np.cos(theta[~is_upper])

    # Apply angle of attack
    xp = xp * np.cos(angle_of_attack) + yp * np.sin(angle_of_attack)
    yp = yp * np.cos(angle_of_attack) - xp * np.sin(angle_of_attack)

    return xp, yp


def wing(
    x,
    # angle between the chord line and the airflow
    angle_of_attack=0.2,
    # angle between the chord line and the x-axis
    sweepback_angle=0.5,
    # angle between the wing and the x-axis
    dihedral_angle=0.08,
    # ratio of the tip chord to the root chord
    taper_ratio=0.45,
    # ratio of the wing's length to its width
    aspect_ratio=5.6,
):
    x = np.array(x, copy=True)

    """
    x: right (to the tip of the wing)
    y: up
    z: backword 
    """

    ls = x[:, 0]  # Spanwise parameter
    ts = x[:, 1]  # Airfoil parameter

    # Get airfoil shape
    zp, yp = airfoil(ts, angle_of_attack=angle_of_attack)

    # Apply aspect ratio (therefore xs is spanwise parameter)
    xs = ls * aspect_ratio

    # Apply taper ratio
    zp = zp * (1 - ls) + zp * ls * taper_ratio
    yp = yp * (1 - ls) + yp * ls * taper_ratio

    # Apply sweepback angle
    zp = zp + xs * np.tan(sweepback_angle)

    # Apply dihedral angle
    yp = yp + xs * np.tan(dihedral_angle)

    return np.column_stack([xs, zp, yp])


def circle(t, r=1, x=0, y=0):
    # t: 0~1
    # angle: from pi, counter-clockwise
    angle = -2 * np.pi * (t + 0.5)
    xs = np.cos(angle) * r + x
    ys = np.sin(angle) * r + y
    return xs, ys


def mount(x):
    x = np.array(x, copy=True)
    ls = x[:, 0]  # Spanwise parameter
    ts = x[:, 1]  # Airfoil parameter

    T = 0.1

    xs = np.zeros_like(ls)
    ls = ls * (1 + T) - T
    xs[ls < 0] = ls[ls < 0]
    ls[ls < 0] = 0
    azp, ayp = airfoil(ts, angle_of_attack=0.2)
    mzp, myp = circle(ts, r=0.6, x=0.5, y=0.05)
    zp = mzp * (1 - ls) + azp * ls
    yp = myp * (1 - ls) + ayp * ls
    return np.column_stack([xs, zp, yp])


def wing_with_mount(x):
    x = np.array(x, copy=True)

    T = 0.5
    is_mount = x[:, 0] <= T

    input_mount = x[is_mount]
    input_mount[:, 0] = input_mount[:, 0] / T

    input_wing = x[~is_mount]
    input_wing[:, 0] = (input_wing[:, 0] - T) / (1 - T)

    part_mount = mount(input_mount)
    part_wing = wing(input_wing)

    result = np.zeros((x.shape[0], 3))
    result[is_mount] = part_mount
    result[~is_mount] = part_wing

    return result
