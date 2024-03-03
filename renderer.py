from matplotlib import pyplot as plt
import numpy as np


def main():
    # Load data from csv files
    nodes = np.loadtxt("nodes.csv", delimiter=",")
    faces = np.loadtxt("faces.csv", delimiter=",", dtype=int)
    vertices = np.loadtxt("vertices.csv", delimiter=",")

    fig = plt.figure()

    # Draw 3D surface
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = vertices[:, 0], vertices[:, 1], vertices[:, 2]
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


if __name__ == "__main__":
    main()
