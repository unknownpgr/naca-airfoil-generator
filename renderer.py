from matplotlib import pyplot as plt
import numpy as np


def main():
    # Load data from csv files
    print("Loading data...")
    nodes = np.loadtxt("output/data-nodes.csv", delimiter=",")
    faces = np.loadtxt("output/data-faces.csv", delimiter=",", dtype=int)
    vertices = np.loadtxt("output/data-vertices.csv", delimiter=",")

    # Set dpi
    plt.rcParams["figure.dpi"] = 300

    # Draw 3D surface
    print("Drawing...")
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ax.plot_trisurf(xs, ys, zs, triangles=faces, cmap="viridis")
    ax.axis("equal")
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    fig.savefig("output/graph-model-default.png", bbox_inches="tight")
    ax.view_init(0, 0)
    fig.savefig("output/graph-model-front.png", bbox_inches="tight")
    ax.view_init(0, 90)
    fig.savefig("output/graph-model-right.png", bbox_inches="tight")
    ax.view_init(90, 0)
    fig.savefig("output/graph-model-top.png", bbox_inches="tight")
    plt.close()

    # Draw 2D scatter
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=0.1)
    ax.axis("equal")
    fig.savefig("output/graph-vertices.png")
    plt.close()


if __name__ == "__main__":
    main()
