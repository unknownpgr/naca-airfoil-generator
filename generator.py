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

    def is_flat(func, face, nodes):
        """
        Test if the function is flat enough on the given surface.
        Return True if the function is flat enough, False otherwise.
        """
        ia, ib, ic = face
        ax, ay = nodes[ia]
        bx, by = nodes[ib]
        cx, cy = nodes[ic]
        a = func(ax, ay)
        b = func(bx, by)
        c = func(cx, cy)
        for i in range(len(weights)):
            w = weights[i]
            sample_x = w[0] * ax + w[1] * bx + w[2] * cx
            sample_y = w[0] * ay + w[1] * by + w[2] * cy
            expected = w[0] * a + w[1] * b + w[2] * c
            actual = func(sample_x, sample_y)
            if np.linalg.norm(actual - expected) > 0.01:
                return False
        return True

    def divide(func, face, nodes, max_depth=10, depth=0):

        if depth == max_depth:
            return nodes, [face]

        if is_flat(func, face, nodes):
            return nodes, [face]

        ia, ib, ic = face
        a = nodes[ia]
        b = nodes[ib]
        c = nodes[ic]
        ab = (a + b) / 2
        bc = (b + c) / 2
        ca = (c + a) / 2
        nodes = np.vstack([nodes, ab, bc, ca])
        ab_idx = len(nodes) - 3
        bc_idx = len(nodes) - 2
        ca_idx = len(nodes) - 1
        sub_faces = [
            [ia, ab_idx, ca_idx],
            [ab_idx, ib, bc_idx],
            [ca_idx, bc_idx, ic],
            [ab_idx, bc_idx, ca_idx],
        ]
        result_faces = []
        for face in sub_faces:
            updated_nodes, sub_faces = divide(func, face, nodes, max_depth, depth + 1)
            nodes = updated_nodes
            result_faces.extend(sub_faces)
        return nodes, result_faces

    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    faces = [[0, 1, 2], [0, 2, 3]]
    result_faces = []
    for face in faces:
        nodes, sub_faces = divide(func, face, nodes, max_depth)
        result_faces.extend(sub_faces)
    return nodes, result_faces


def airfoil(x, m=0.1, p=0.6, t=0.15, c=1):
    """
    Define the airfoil shape using the NACA airfoil.
    """

    is_upper = x < 0.5
    if is_upper:
        x *= 2
    else:
        x = 2 * (1 - x)

    # Camber line
    if x < p:
        yc = m / p**2 * (2 * p * x - x**2)
        dyc = 2 * m / p**2 * (p - x)
    else:
        yc = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2)
        dyc = 2 * m / (1 - p) ** 2 * (p - x)
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

    if is_upper:
        # Upper surface
        xp = x - yt * np.sin(theta)
        yp = yc + yt * np.cos(theta)
    else:
        # Lower surface
        xp = x + yt * np.sin(theta)
        yp = yc - yt * np.cos(theta)

    return xp, yp


def wing(t, x):
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

    # Get airfoil shape
    zp, yp = airfoil(t)

    # Apply taper ratio
    zp = zp * (1 - x) + zp * x * taper_ratio
    yp = yp * (1 - x) + yp * x * taper_ratio

    # Apply aspect ratio
    x = x * aspect_ratio

    # Apply angle of attack
    zp = zp * np.cos(angle_of_attack) + yp * np.sin(angle_of_attack)
    yp = yp * np.cos(angle_of_attack) - zp * np.sin(angle_of_attack)

    # Apply sweepback angle
    zp = zp + x * np.tan(sweepback_angle)
    yp = yp

    # Apply dihedral angle
    zp = zp
    yp = yp + x * np.tan(dihedral_angle)

    return np.array([zp, x, yp]) * 20


def __main__():
    nodes, faces = model(wing, max_depth=10)
    fig = plt.figure()

    # Map 2D to 3D
    points = [wing(nodes[i, 0], nodes[i, 1]) for i in range(len(nodes))]
    points = np.array(points)

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
    plt.savefig("output-model.png")
    plt.close()

    # Draw 2D scatter
    plt.scatter(nodes[:, 0], nodes[:, 1], s=1)
    plt.axis("equal")
    plt.savefig("output-vertices.png")
    plt.close()


__main__()
