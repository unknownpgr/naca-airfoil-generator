from matplotlib import pyplot as plt
import numpy as np


def model(func, max_depth=10):
    weights = [
        # Points
        # [1, 0, 0],
        # [0, 1, 0],
        # [0, 0, 1],
        # Midpoints
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        # Centers
        [1, 1, 1],
        [1, 3, 1],
        [3, 1, 1],
        [1, 1, 3],
    ]
    weights = np.array(weights, dtype=float)
    weights /= weights.sum(axis=1)[:, None]

    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    faces = []
    queue = [(0, [0, 1, 2]), (0, [0, 2, 3])]

    def is_flat(face):
        """
        Test if the function is flat enough on the given surface.
        Return True if the function is flat enough, False otherwise.
        """
        vertices = nodes[face]
        outputs = func(weights @ vertices)
        expects = weights @ func(vertices)
        diff = np.linalg.norm(outputs - expects, axis=1)
        return np.all(diff < 0.01)

    while queue:
        depth, face = queue.pop()
        if depth == max_depth or is_flat(face):
            faces.append(face)
            continue

        ia, ib, ic = face
        points = nodes[face]
        rotated_points = np.roll(points, -1, axis=0)
        new_nodes = (points + rotated_points) / 2
        iab = len(nodes)
        ibc = iab + 1
        ica = ibc + 1
        nodes = np.vstack([nodes, new_nodes])

        sub_faces = [
            [ia, iab, ica],
            [iab, ib, ibc],
            [ica, ibc, ic],
            [iab, ibc, ica],
        ]
        for sub_face in sub_faces:
            queue.append((depth + 1, sub_face))

    return nodes, faces


def airfoil(x, m=0.1, p=0.6, t=0.15, c=1):
    """
    Define the airfoil shape using the NACA airfoil.
    """

    is_upper = x < 0.5
    x[is_upper] = x[is_upper] * 2
    x[~is_upper] = 2 * (1 - x[~is_upper])

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
        * c
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


def __main__():
    nodes, faces = model(wing, max_depth=10)
    print(f"Number of vertices: {len(nodes)}")
    print(f"Number of faces: {len(faces)}")

    fig = plt.figure()

    # Map 2D to 3D
    points = wing(nodes)

    # Save to obj file
    with open("output.obj", "w") as f:
        for i in range(len(points)):
            f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
        for i in range(len(faces)):
            f.write(f"f {faces[i][0] + 1} {faces[i][1] + 1} {faces[i][2] + 1}\n")

    # Draw 3D surface
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.plot_trisurf(xs, ys, zs, triangles=faces, cmap="viridis")
    plt.axis("equal")
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    plt.savefig("output-model-default.png")
    ax.view_init(0, 0)
    plt.savefig("output-model-front.png")
    ax.view_init(0, 90)
    plt.savefig("output-model-right.png")
    ax.view_init(90, 0)
    plt.savefig("output-model-top.png")
    plt.close()

    # Draw 2D scatter
    plt.scatter(nodes[:, 0], nodes[:, 1], s=1)
    plt.axis("equal")
    plt.savefig("output-vertices.png")
    plt.close()


__main__()
