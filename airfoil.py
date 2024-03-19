import numpy as np


def airfoil(x):
    """
    Define the airfoil shape using the NACA airfoil.
    """

    x = np.array(x, copy=True)

    m = 0.1
    p = 0.6
    t = 0.15

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

    return xp, yp


def wing(x):
    x = np.array(x, copy=True)

    """
    x: right
    y: up
    z: backword
    """
    angle_of_attack = 0.2  # 받음각 - x축 기준 날개 단면의 각도 (~15도)
    sweepback_angle = 0.5  # 후퇴각 - y축 기준 날개가 x축과 이루는 각도
    dihedral_angle = 0.08  # 상반각 - z축 기준 날개가 x축과 이루는 각도 (~4.5도)
    taper_ratio = 0.45  # 테이퍼율 - 본체 쪽과 끝쪽의 폭 비율
    aspect_ratio = 5.6  # 가로세로비 - 날개의 길이와 폭의 비율

    ts = x[:, 0]  # Airfoil parameter
    ls = x[:, 1]  # Spanwise parameter

    # Get airfoil shape
    zp, yp = airfoil(ts)

    # Apply aspect ratio (therefore xs is spanwise parameter)
    xs = ls * aspect_ratio

    # Apply taper ratio
    zp = zp * (1 - ls) + zp * ls * taper_ratio
    yp = yp * (1 - ls) + yp * ls * taper_ratio

    # Apply angle of attack
    zp = zp * np.cos(angle_of_attack) + yp * np.sin(angle_of_attack)
    yp = yp * np.cos(angle_of_attack) - zp * np.sin(angle_of_attack)

    # Apply sweepback angle
    zp = zp + xs * np.tan(sweepback_angle)

    # Apply dihedral angle
    yp = yp + xs * np.tan(dihedral_angle)

    return np.column_stack([xs, zp, yp])


def base(t, r=1):
    angle = -(t + 0.5) * 2 * np.pi
    zp = np.cos(angle) * r + 0.5
    yp = np.sin(angle) * r + 0.05
    return zp, yp


def mount(x):
    x = np.array(x, copy=True)
    ts = x[:, 0]  # Airfoil parameter
    ls = x[:, 1]  # Spanwise parameter

    T = 0.1

    xs = np.zeros_like(ls)
    ls = ls * (1 + T) - T
    xs[ls < 0] = ls[ls < 0]
    ls[ls < 0] = 0
    azp, ayp = base(ts, r=0.5)
    mzp, myp = base(ts)
    zp = mzp * (1 - ls) + azp * ls
    yp = myp * (1 - ls) + ayp * ls
    return np.column_stack([xs, zp, yp])


def wing_with_mount(x):
    x = np.array(x, copy=True)

    T = 0.5
    is_mount = x[:, 1] <= T

    input_mount = x[is_mount]
    input_mount[:, 1] = input_mount[:, 1] / T

    input_wing = x[~is_mount]
    input_wing[:, 1] = (input_wing[:, 1] - T) / (1 - T)

    part_mount = mount(input_mount)
    part_wing = wing(input_wing)

    result = np.zeros((x.shape[0], 3))
    result[is_mount] = part_mount
    result[~is_mount] = part_wing

    return result
