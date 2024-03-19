import numpy as np
from model import wing_with_mount
from modeler import Modeler
from matplotlib import pyplot as plt

print("Start modeling")
modeler = Modeler()
model = modeler.model(wing_with_mount, max_depth=7, threshold=0.005)
model.close_edge(model.initial_left)
model.close_edge(model.initial_right, reverse_face=True)
model.weave_edges(model.initial_bottom, model.initial_top)
print("Done.\n")

print("Writing to file")
with open("output/model.obj", "w") as f:
    obj_str = model.get_obj_str()
    f.write(obj_str)
print("Done.\n")

print("Visualizing points")
plt.plot(
    model.points[:, 0],
    model.points[:, 1],
    "o",
    color="black",
    markersize=1,
)
plt.savefig("output/points.png")
plt.clf()

print("Visualizing 2D mesh")
plt.triplot(
    model.points[:, 0],
    model.points[:, 1],
    model.faces,
    color="black",
    linewidth=0.1,
)
plt.savefig("output/mesh.png")
plt.clf()

print("Visualizing test points")
initial_points = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
initial_points = np.array(initial_points)
weights = modeler.get_weights()
weighted_points = weights @ initial_points
plt.plot(
    initial_points[:, 0],
    initial_points[:, 1],
    "o",
    color="red",
    markersize=2,
)
plt.plot(
    weighted_points[:, 0],
    weighted_points[:, 1],
    "o",
    color="black",
    markersize=1,
)
plt.savefig("output/test_points.png")
plt.clf()

print("Visualizing 3D model")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])
ax.set_proj_type("ortho")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.plot_trisurf(
    model.vertices[:, 0],
    model.vertices[:, 1],
    model.vertices[:, 2],
    triangles=model.faces,
    color="white",
    edgecolor="black",
    linewidth=0.1,
)
# Add axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.axis("equal")
i = 0
for angle in range(0, 181, 45):
    i += 1
    filename = f"output/model-{i}.png"
    print(f"Saving {filename}")
    ax.view_init(elev=45, azim=angle - 90)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=300)
