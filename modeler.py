import numpy as np
from dataclasses import dataclass
from typing import List
from matplotlib import pyplot as plt
import time


@dataclass
class Model:
    points: np.ndarray
    faces: np.ndarray
    vertices: np.ndarray
    initial_left: List[int]
    initial_right: List[int]
    initial_bottom: List[int]
    initial_top: List[int]

    def get_obj_str(self):
        obj = ""
        for v in self.vertices:
            obj += f"v {v[0]} {v[1]} {v[2]}\n"
        for f in self.faces:
            obj += f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n"
        return obj


class Modeler:

    @staticmethod
    def weave_edges(model, edge1, edge2, reverse_face=False):
        vs1 = model.vertices[edge1]
        vs2 = model.vertices[edge2]

        new_faces = []
        i1 = 0
        i2 = 0

        norm = lambda v: np.linalg.norm(v)

        while i1 < len(vs1) - 1 or i2 < len(vs2) - 1:
            if i1 == len(vs1) - 1:
                new_faces.append([edge1[i1], edge2[i2], edge2[i2 + 1]])
                i2 += 1
                continue
            if i2 == len(vs2) - 1:
                new_faces.append([edge1[i1], edge2[i2], edge1[i1 + 1]])
                i1 += 1
                continue
            if norm(vs1[i1] - vs2[i2]) < norm(vs1[i1 + 1] - vs2[i2]):
                new_faces.append([edge1[i1], edge2[i2], edge2[i2 + 1]])
                i2 += 1
            else:
                new_faces.append([edge1[i1], edge2[i2], edge1[i1 + 1]])
                i1 += 1

        if reverse_face:
            new_faces = [f[::-1] for f in new_faces]

        model.faces = np.vstack([model.faces, new_faces])

    @staticmethod
    def close_edge(model, edge, reverse_face=False):
        if len(edge) < 3:
            raise ValueError("Cannot close an edge with less than 3 vertices")

        if len(edge) == 3:
            new_faces = []
            if reverse_face:
                new_faces.append([edge[0], edge[2], edge[1]])
            else:
                new_faces.append([edge[0], edge[1], edge[2]])
            model.faces = np.vstack([model.faces, new_faces])
            return

        n = len(edge)
        left = edge[: n // 2]
        right = edge[n // 2 :]
        left = left[::-1]
        Modeler.weave_edges(model, left, right, reverse_face=reverse_face)

    def __init__(self):
        test_weights = [
            # Point
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            # Somewhere in the square
            [5, 1, 1, 1],
            [1, 5, 1, 1],
            [1, 1, 5, 1],
            [1, 1, 1, 5],
            # Middle of the edge
            [4, 4, 1, 1],
            [4, 1, 4, 1],
            [1, 4, 1, 4],
            [1, 1, 4, 4],
            # Center
            [1, 1, 1, 1],
        ]
        test_weights = np.array(test_weights, dtype=np.float32)
        test_weights /= test_weights.sum(axis=1, keepdims=True)
        self.__test_weights = test_weights
        self.__points = []

    def __initialize(self):
        self.__points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    def __divide_edge(self, edge):
        i1, i2, children = edge
        if len(children) == 2:
            return children
        p1 = self.__points[i1]
        p2 = self.__points[i2]
        pm = (p1 + p2) / 2
        self.__points = np.vstack([self.__points, pm])
        im = len(self.__points) - 1

        c1 = [i1, im, []]
        c2 = [im, i2, []]
        children.append(c1)
        children.append(c2)
        return [c1, c2]

    def __test_flatness(self, rect, func):
        """
        Test the flatness (linearity) of the given rect.
        """
        test_inputs = self.__test_weights @ rect
        test_outputs = func(test_inputs)

        """
        The transformed shape should be planar.
        It means that there exists a plane that contains all the points.
        
        A plane is determined by center point and normal vector.
        We can roughly assume that the center point is the average of the
        first four points, and the normal vector is the cross product of diagonals.
        """

        ps = test_outputs[:4]
        center = ps.mean(axis=0)
        normal = np.cross(ps[0] - ps[3], ps[1] - ps[2])
        normal /= np.linalg.norm(normal)

        """
        Before calculating, for the ease of calculation, we can move the center to the origin
        and rotate the normal vector to the z-axis.
        """

        test_outputs -= center
        original_z = normal
        original_x = ps[0] - ps[3]
        original_x /= np.linalg.norm(original_x)
        original_y = np.cross(original_z, original_x)
        original_basis = np.vstack([original_x, original_y, original_z])
        original_basis_inv = np.linalg.inv(original_basis)
        test_outputs = test_outputs @ original_basis_inv

        """
        Before calculating the distance, we must check that the transformed shape is convex.
        It means that the other points can be represented as the convex combination of the first four points.
        we should flatten the points to the plane because the points are not on same plane in general.
        """

        flattened_points = test_outputs[:, :2]
        flattened_points -= flattened_points[0]
        basis = flattened_points[1:3]
        test_points = flattened_points[4:]
        a = np.linalg.lstsq(basis.T, test_points.T, rcond=None)[0].T
        if np.any(a < 0) or np.any(a > 1):
            return np.inf

        """
        Because the plane is now the z=0 plane, the distance of the points to the plane
        is simply the z-coordinate of the points.
        """

        distances = test_outputs[:, 2]

        """
        The flatness of the shape is the maximum distance of the points to the plane.
        """

        return np.max(np.abs(distances))

    def model(self, func, max_depth=7, threshold=0.05):
        self.__initialize()
        first_face = [
            (0, 1, []),  # Left
            (2, 3, []),  # Right
            (0, 2, []),  # Bottom
            (1, 3, []),  # Top
        ]
        stack = [(0, first_face)]
        faces = []

        while stack:
            depth, face = stack.pop()
            if depth == max_depth:
                faces.append(face)
                continue

            indexes = [face[0][0], face[0][1], face[1][0], face[1][1]]
            rect = self.__points[indexes]
            diff = self.__test_flatness(rect, func)
            if diff < threshold:
                faces.append(face)
                continue

            mid = (rect[0] + rect[3]) / 2
            self.__points = np.vstack([self.__points, mid])
            im = len(self.__points) - 1

            lb, lt = self.__divide_edge(face[0])
            rb, rt = self.__divide_edge(face[1])
            bl, br = self.__divide_edge(face[2])
            tl, tr = self.__divide_edge(face[3])

            ml = (lb[1], im, [])
            mr = (im, rb[1], [])
            mb = (bl[1], im, [])
            mt = (im, tl[1], [])

            stack.append((depth + 1, [lb, mb, bl, ml]))
            stack.append((depth + 1, [lt, mt, ml, tl]))
            stack.append((depth + 1, [mb, rb, br, mr]))
            stack.append((depth + 1, [mt, rt, mr, tr]))

        result_faces = []

        def expand_edge(edge):
            i1, i2, children = edge
            if not children:
                return [i1, i2]
            return expand_edge(children[0])[:-1] + expand_edge(children[1])

        def expand_face(face):
            e1 = expand_edge(face[0])  # Left
            e2 = expand_edge(face[1])  # Right
            e3 = expand_edge(face[2])  # Bottom
            e4 = expand_edge(face[3])  # Top
            et = e3 + e2 + e4[::-1] + e1[::-1]
            eu = [et[0]]
            for i in range(1, len(et)):
                if et[i] != et[i - 1]:
                    eu.append(et[i])
            return eu

        def triangulate(face):
            f = expand_face(face)
            assert f[0] == f[-1]
            f = f[:-1]
            n = len(f)

            if n < 3:
                raise ValueError("Cannot triangulate a face with less than 3 vertices")

            if n == 3:
                result_faces.append(f)
                return

            if n == 4:
                result_faces.append([f[0], f[1], f[2]])
                result_faces.append([f[0], f[2], f[3]])
                return

            # Triangulate the face
            center = self.__points[f].mean(axis=0)
            ci = len(self.__points)
            self.__points = np.vstack([self.__points, center])

            for i in range(n):
                result_faces.append([f[i], f[(i + 1) % n], ci])

        for face in faces:
            triangulate(face)

        vertices = func(self.__points)

        return Model(
            points=self.__points,
            faces=result_faces,
            vertices=vertices,
            initial_left=expand_edge(first_face[0]),
            initial_right=expand_edge(first_face[1]),
            initial_bottom=expand_edge(first_face[2]),
            initial_top=expand_edge(first_face[3]),
        )

    def get_weights(self):
        return self.__test_weights
