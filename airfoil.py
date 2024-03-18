import numpy as np
from modeler import Modeler
from matplotlib import pyplot as plt


def airfoil(x):
    """
    Define the airfoil shape using the NACA airfoil.
    """

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
    yp[is_upper] += 1

    return xp, yp


def wing(x):
    x = np.array(x)

    """
    x: right
    y: up
    z: backword
    """
    angle_of_attack = 0.2  # 받음각 - x축 기준 날개 단면의 각도
    sweepback_angle = 0.5  # 후퇴각 - y축 기준 날개가 x축과 이루는 각도
    dihedral_angle = 0.13  # 상반각 - z축 기준 날개가 x축과 이루는 각도
    taper_ratio = 0.45  # 테이퍼율 - 본체 쪽과 끝쪽의 폭 비율
    aspect_ratio = 5.6  # 가로세로비 - 날개의 길이와 폭의 비율

    ts = x[:, 0]  # Airfoil parameter
    ls = x[:, 1]  # Spanwise parameter

    # Get airfoil shape
    zp, yp = airfoil(ts)

    # Apply aspect ratio
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

    return np.column_stack([zp, xs, yp]) * 20


modeler = Modeler()
model = modeler.model(wing, max_depth=8, threshold=0.01)

Modeler.close_edge(model, model.initial_bottom, reverse_face=True)
Modeler.close_edge(model, model.initial_top)
Modeler.weave_edges(model, model.initial_right, model.initial_left)

with open("output/model.obj", "w") as f:
    obj_str = model.get_obj_str()
    f.write(obj_str)

faces = model.faces
vertices = model.vertices

fig = plt.figure()
fig.set_size_inches(50, 50)
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=faces,
    color="gray",
    edgecolor="black",
)
plt.savefig("output/airfoil-1.png")
