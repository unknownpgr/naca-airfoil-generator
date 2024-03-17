"""
기존의 generator 방식은 메시를 잘 나누기는 했으나 그 토폴로지가 잘 보존되지 않았다.
이에 메시를 트리 구조로 나누어 토폴로지를 보존할 수 있도록 수정한다.

먼저 이전의 삼각형과 다르게 메시를 사각형으로 구성, 이전과 마찬가지로 사각형이 충분히 평평해질 때까지 sub-deviding을 반복할 것이다.
이 방법은 도메인의 모양이 사각형이므로 직관적으로 이해하기 쉽고 이후에 각 사각형을 삼각형으로 나누기만 하면 쉽게 face들을 구성할 수 있다는 장점이 있다.
"""

import numpy as np
from matplotlib import pyplot as plt

test_weights = [
    # Point
    # [1, 0, 0, 0],
    # [0, 1, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1],
    # Somewhere in the square
    [5, 1, 1, 1],
    [1, 5, 1, 1],
    [1, 1, 5, 1],
    [1, 1, 1, 5],
    # Middle of the edge
    [1, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    # Center
    [1, 1, 1, 1],
]
test_weights = np.array(test_weights, dtype=np.float32)
test_weights /= test_weights.sum(axis=1, keepdims=True)


def test_surface(rect, func):
    test_inputs = test_weights @ rect
    test_outputs = func(test_inputs)

    output_points = func(rect)
    weighted_output = test_weights @ output_points

    diff = test_outputs - weighted_output
    diff = np.linalg.norm(diff, axis=1)
    return diff


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

    return xp, yp


def wing(x):
    """
    x: right
    y: up
    z: backword
    """
    angle_of_attack = 0.2  # 받음각 - x축 기준 날개 단면의 각도
    sweepback_angle = 0.5  # 후퇴각 - y축 기준 날개가 x축과 이루는 각도
    dihedral_angle = 0.13  # 상반각 - z축 기준 날개가 x축과 이루는 각도
    taper_ratio = 0.4  # 테이퍼율 - 본체 쪽과 끝쪽의 폭 비율
    aspect_ratio = 5.6 / 2  # 가로세로비 - 날개의 길이와 폭의 비율

    # Deep clone x
    x = np.array(x)

    ts = x[:, 0]  # Airfoil parameter
    ls = x[:, 1]  # Spanwise parameter

    # Get airfoil shape
    zp, yp = airfoil(ts)

    # Increase ls for the tip
    ls = ls * 1.2 - 0.1

    # Apply aspect ratio
    xs = ls * aspect_ratio

    # Apply taper ratio
    zp = zp * (1 - ls) + zp * ls * taper_ratio
    yp = yp * (1 - ls) + yp * ls * taper_ratio

    # Close tips
    zp[ls < 0] *= 1 + ls[ls < 0] * 10
    yp[ls < 0] *= 1 + ls[ls < 0] * 10
    yp[ls > 1] *= 1 - (ls[ls > 1] - 1) * 10
    zp[ls > 1] *= 1 - (ls[ls > 1] - 1) * 10
    xs[ls < 0] = 0
    xs[ls > 1] = aspect_ratio

    # Apply angle of attack
    zp = zp * np.cos(angle_of_attack) + yp * np.sin(angle_of_attack)
    yp = yp * np.cos(angle_of_attack) - zp * np.sin(angle_of_attack)

    # Apply sweepback angle
    zp = zp + xs * np.tan(sweepback_angle)

    # Apply dihedral angle
    yp = yp + xs * np.tan(dihedral_angle)

    return np.column_stack([zp, xs, yp]) * 20


def surface_func(v):
    ts = v[:, 0]
    ys = v[:, 1]
    return np.stack(
        [np.sin(ts * 20) * ts * (ys + 2), np.cos(ts * 20) * ts * (ys + 2), ys], axis=1
    )


surface_func = wing

points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

"""
Edge의 방향은 반드시 작은 쪽에서 큰 쪽으로 가야 한다.
"""


def devide_edge(edge):
    global points

    i1, i2, children = edge
    if len(children) == 2:
        return children
    p1 = points[i1]
    p2 = points[i2]
    pm = (p1 + p2) / 2
    points = np.vstack([points, pm])
    im = len(points) - 1

    c1 = [i1, im, []]
    c2 = [im, i2, []]
    children.append(c1)
    children.append(c2)
    return [c1, c2]


first_face = [
    # Left
    (0, 1, []),
    # Right
    (2, 3, []),
    # Bottom
    (0, 2, []),
    # Top
    (1, 3, []),
]
stack = [(0, first_face)]
faces = []
while stack:
    depth, face = stack.pop()
    if depth == 8:
        faces.append(face)
        continue

    indexes = [face[0][0], face[0][1], face[1][0], face[1][1]]
    rect = points[indexes]
    diff = test_surface(rect, surface_func)
    if np.all(diff < 0.001):
        faces.append(face)
        continue

    mid = (rect[0] + rect[3]) / 2
    points = np.vstack([points, mid])
    im = len(points) - 1

    lb, lt = devide_edge(face[0])
    rb, rt = devide_edge(face[1])
    bl, br = devide_edge(face[2])
    tl, tr = devide_edge(face[3])

    ml = (lb[1], im, [])
    mr = (im, rb[1], [])
    mb = (bl[1], im, [])
    mt = (im, tl[1], [])

    stack.append((depth + 1, [lb, mb, bl, ml]))
    stack.append((depth + 1, [lt, mt, ml, tl]))
    stack.append((depth + 1, [mb, rb, br, mr]))
    stack.append((depth + 1, [mt, rt, mr, tr]))

print("Dividing finished.")

vertices = surface_func(points)
result_faces = []


def expand_edge(edge):
    i1, i2, children = edge
    if not children:
        return [i1, i2]
    return expand_edge(children[0]) + expand_edge(children[1])


def expand_face(face):
    e1 = expand_edge(face[0])  # Left
    e2 = expand_edge(face[1])  # Right
    e3 = expand_edge(face[2])  # Bottom
    e4 = expand_edge(face[3])  # Top
    return e1[:-1] + e4[:-1] + list(reversed(e2))[:-1] + list(reversed(e3))


def triangulate(points):
    faces = []
    for i in range(1, len(points) - 1):
        faces.append([points[0], points[i], points[i + 1]])
    return faces


result_faces = []
for face in faces:
    result_faces.extend(triangulate(expand_face(face)))

result_faces = np.array(result_faces, dtype=np.int32)

print("Triangulation finished.")

with open("model.obj", "w") as f:
    # f.write("# Vertices\n")
    for v in vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    # f.write("# Faces\n")
    for face in result_faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

print("Model saved.")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=result_faces,
    cmap="viridis",
)
plt.savefig("test.png")
plt.clf()

plt.axis("equal")
# plt.triplot(points[:, 0], points[:, 1], result_faces,
plt.plot(points[:, 0], points[:, 1], "o", markersize=1)
plt.savefig("test2.png")
plt.clf()

print("Done")
