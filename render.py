from airfoil import wing_with_mount
from modeler import Modeler
from matplotlib import pyplot as plt
import os

print("Start modeling")
modeler = Modeler()
model = modeler.model(wing_with_mount, max_depth=8, threshold=0.005)
Modeler.close_edge(model, model.initial_bottom, reverse_face=True)
Modeler.close_edge(model, model.initial_top)
Modeler.weave_edges(model, model.initial_right, model.initial_left)
print("Done.\n")

print("Writing to file")
with open("output/model.obj", "w") as f:
    obj_str = model.get_obj_str()
    f.write(obj_str)
print("Done.\n")

print("Visualizing")
plt.plot(
    model.points[:, 0],
    model.points[:, 1],
    "o",
    color="black",
    markersize=1,
)
plt.savefig("output/points.png")
plt.clf()

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
    ax.view_init(elev=45, azim=angle - 90)
    fig.savefig(f"output/airfoil-{i}.png", bbox_inches="tight", pad_inches=0, dpi=300)
