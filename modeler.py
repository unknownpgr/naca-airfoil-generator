# 기존의 generator 방식은 메시를 잘 나누기는 했으나 그 토폴로지가 잘 보존되지 않았다.
# 이에 메시를 트리 구조로 나누어 토폴로지를 보존할 수 있도록 수정한다.
# 먼저 이전의 삼각형과 다르게 메시를 사각형으로 구성, 이전과 마찬가지로 사각형이 충분히 평평해질 때까지 sub-deviding을 반복할 것이다.
# 이 방법은 도메인의 모양이 사각형이므로 직관적으로 이해하기 쉽고 이후에 각 사각형을 삼각형으로 나누기만 하면 쉽게 face들을 구성할 수 있다는 장점이 있다.
# 사각형을 구성하는 점의 순서는 LB, LT, RB, RT로 한다.
# 모든 에지는 반드시 값이 낮은 쪽에서 높은 쪽으로 향하도록 한다.

import numpy as np
from dataclasses import dataclass
from typing import List


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
    def __expand_edge(edge):
        i1, i2, children = edge
        if not children:
            return [i1, i2]
        return Modeler.__expand_edge(children[0])[:-1] + Modeler.__expand_edge(
            children[1]
        )

    @staticmethod
    def __expand_face(face):
        e1 = Modeler.__expand_edge(face[0])  # Left
        e2 = Modeler.__expand_edge(face[1])  # Right
        e3 = Modeler.__expand_edge(face[2])  # Bottom
        e4 = Modeler.__expand_edge(face[3])  # Top
        et = e3 + e2 + e4[::-1] + e1[::-1]
        eu = [et[0]]
        for i in range(1, len(et)):
            if et[i] != et[i - 1]:
                eu.append(et[i])
        return eu

    @staticmethod
    def __triangulate(points):
        if len(points) == 3:
            raise ValueError("Cannot triangulate a triangle")
        if len(points) == 4:
            return [
                [points[0], points[1], points[2]],
                [points[0], points[2], points[3]],
            ]
        triangles = []
        for i in range(1, len(points) - 1):
            triangles.append([points[0], points[i], points[i + 1]])
        return triangles

    @staticmethod
    def weave_edges(model, edge1, edge2):
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
        if reverse_face:
            left, right = right, left
        left = left[::-1]
        Modeler.weave_edges(model, left, right)

    def __init__(self):
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
        주어진 rect가 얼마나 flat한지 테스트한다.
        평면 근사를 통한 판정을 포함한 여러 방법을 테스트해봤지만,
        이 방법이 제일 낫다.
        평면 판정의 경우 심하게 구부러진 도형을 평면으로 판정하는 경우가 많았다.
        """
        test_inputs = self.__test_weights @ rect
        test_outputs = func(test_inputs)

        output_points = func(rect)
        weighted_output = self.__test_weights @ output_points

        diff = test_outputs - weighted_output
        diff = np.linalg.norm(diff, axis=1)
        return np.max(diff)

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
        for face in faces:
            result_faces.extend(Modeler.__triangulate(Modeler.__expand_face(face)))
        result_faces = np.array(result_faces, dtype=np.int32)

        vertices = func(self.__points)

        return Model(
            points=self.__points,
            faces=result_faces,
            vertices=vertices,
            initial_left=Modeler.__expand_edge(first_face[0]),
            initial_right=Modeler.__expand_edge(first_face[1]),
            initial_bottom=Modeler.__expand_edge(first_face[2]),
            initial_top=Modeler.__expand_edge(first_face[3]),
        )
