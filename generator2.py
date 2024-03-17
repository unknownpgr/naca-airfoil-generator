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
    diff = np.linalg.norm(diff, axis=1).mean()
    return diff


def surface_func(v):
    xs = v[:, 0]
    ys = v[:, 1]
    return np.stack([xs, ys, np.sin((xs**2 + ys**2) * 2)], axis=1)


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
    if depth == 5:
        faces.append(face)
        continue

    indexes = [face[0][0], face[0][1], face[1][0], face[1][1]]
    rect = points[indexes]
    diff = test_surface(rect, surface_func)
    if diff < 0.01:
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

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=result_faces)
plt.savefig("test.png")
plt.clf()

plt.triplot(vertices[:, 0], vertices[:, 1], result_faces)
plt.plot(vertices[:, 0], vertices[:, 1], "o")
plt.savefig("test2.png")
plt.clf()

print("Done")
