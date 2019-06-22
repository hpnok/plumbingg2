from typing import List

import numpy as np
import triangle as tr


class CyclicList(list):
    """
    Element access handle values bigger then len() of the object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except IndexError:
            return super().__getitem__(item%self.__len__())


class VerticesList(list):
    """
    List of vertex to construct iteratively
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def replace_last(self, _t: object):
        try:
            self[-1] = _t
        except IndexError:
            self.append(_t)

    def last_vertex_added(self):
        return self[-1]


class OrthogonalVertexList(object):
    def __init__(self, vertices: np.ndarray):
        """
        :param vertices: Array of shape (n, 1, 2)
        """
        super().__init__()
        self._vertices = vertices
        self.size = len(vertices)
        if self.size == 0:
            raise ValueError("No vertex passed")

    def __getitem__(self, item):
        try:
            return self._vertices[item][0]
        except IndexError:
            return self._vertices[item%self.size][0]

    def __iter__(self):
        for v in self._vertices:
            yield v[0]


class PolygonContour(object):
    """
    The perimeter of a polygon
    """

    def __init__(self, pts: np.ndarray):
        if len(pts) < 3:
            raise ValueError('A polygon must contains at least 3 points')
        pts.resize((len(pts), 2))
        self._vertices = pts
        self._contained_point = None

    def __getitem__(self, item):
        return self._vertices.__getitem__(item)

    def __iter__(self):
        return iter(self._vertices)

    def __len__(self):
        return len(self._vertices)

    @property
    def vertices(self):
        return self._vertices

    @property
    def indices(self):
        return list(range(len(self._vertices)))

    def compute_winding(self):
        previous_pt = self._vertices[-1]
        sum = 0
        for p in self._vertices:
            sum += previous_pt[0]*p[1] - previous_pt[1]*p[0]
            previous_pt = p
        return np.sign(sum)

    @property
    def interior_point(self):
        if self._contained_point is None:
            self._contained_point = self._find_interior_position()
        return self._contained_point

    def _find_interior_position(self) -> np.ndarray:
        target = sum(self._vertices)/len(self._vertices)
        return self._find_interior_position_with_ray_march(target)

    def _find_interior_position_with_ray_march(self, target):
        crossed_left_segments = []
        crossed_right_segments = []

        previous_x, previous_y = self._vertices[-1]
        target_x, target_y = target
        for current_x, current_y in self._vertices:
            if (previous_y > target_y) != (current_y > target_y):
                x_intersection = ((current_x - previous_x)/(current_y - previous_y))*(target_y - previous_y) + previous_x
                if x_intersection < target_x:
                    crossed_left_segments.append(x_intersection)
                else:
                    crossed_right_segments.append(x_intersection)

        if crossed_left_segments:
            if len(crossed_left_segments)%2 == 0:
                crossed_left_segments.sort()
                target[0] = sum(crossed_left_segments[-2:])/2
        elif crossed_right_segments:
            # since the sum of all crossing must be even, we can assume the right set is even
            crossed_right_segments.sort()
            target[0] = sum(crossed_right_segments[:2])/2
        else:
            raise RuntimeError("Could't find a point inside the hole, write some unit test :^)")
        return target


class Polygon(object):
    """
    A polygon triangulated
    """

    # TODO: rename to TriangulatedPolygon, create a Polygon class to handle geomtry operations?
    # TODO: possible correction, we apply a SCALE to every point, then smear the image, the displaced vertices are then on "subpixels" of
    # the original image. By removing those subpixels we undo the smearing??? (must be done before sloping???)
    def __init__(self, pts: np.ndarray, holes: List[np.ndarray] = None):
        self.contour = PolygonContour(pts)
        self.holes = [PolygonContour(hole) for hole in holes]
        input_tri_dict = self._get_tri_dict()

        tri_dict = tr.triangulate(input_tri_dict, 'p')

        self._plane_triangles = tri_dict['triangles']
        self._face_vertices = tri_dict['vertices']
        self._winding = self.contour.compute_winding()

    def _get_tri_dict(self):
        """return dict to pass to triangles lib"""
        tri_dict = dict(
            vertices=np.concatenate([self.contour.vertices] + [hole.vertices for hole in self.holes]),
            segments=list(self._segment_pairs())
        )
        if self.holes:
            tri_dict['holes'] = np.array([hole.interior_point for hole in self.holes])
        return tri_dict

    def face(self, item):
        return self._face_vertices[item]

    def triangles(self):
        return self._plane_triangles

    @property
    def winding(self):
        return self._winding

    def _segment_pairs(self):
        n = len(self.contour)
        for i in range(n):
            yield (i, (i + 1)%n)
        for hole in self.holes:
            m = len(hole)
            for i in range(m):
                yield (n + i, n + (i + 1)%m)
            n += m
