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


class Polygon(object):
    # TODO: possible correction, we apply a SCALE to every point, then smear the image, the displaced vertices are then on "subpixels" of
    # the original image. By removing those subpixels we undo the smearing??? (must be done before sloping???)
    def __init__(self, pts: np.ndarray, holes: List[np.ndarray] = None):
        n = len(pts)
        perimiter_n = n
        if n < 3:
            raise ValueError('A polygon must contains at least 3 points')
        pts.resize((n, 2))
        s = [(i, (i + 1)%n) for i in range(n)]
        self.hole_pts = []
        for hole in holes:
            s, n, hole_pt = self._add_hole(n, s, hole)
            self.hole_pts.append(hole_pt)
        s = np.array(s)
        poly_dict = dict(vertices=pts, segments=s)
        if holes:
            pts = np.concatenate([pts] + holes)
            poly_dict['vertices'] = pts
            poly_dict['holes'] = np.array(self.hole_pts)
        tri_dict = tr.triangulate(poly_dict, 'p')
        self._plane_triangles = tri_dict['triangles']
        self._face_vertices = tri_dict['vertices']
        self._side_perimeter = pts
        self._winding = self._compute_winding(pts[:perimiter_n])
        self._segments = s

    def _add_hole(self, current_vertex_count, segments, hole_vertices):
        n = len(hole_vertices)
        if n < 3:
            raise ValueError('A polygon hole must contains at least 3 points')
        hole_vertices.resize((n, 2))
        s = [(current_vertex_count + i, current_vertex_count + (i + 1)%n) for i in range(n)]  # reverse winding??
        hole_pt = self._find_hole_position(hole_vertices)
        return segments + s, current_vertex_count + n, hole_pt

    def _find_hole_position(self, hole_vertices):
        n = len(hole_vertices)
        hole_pt = sum(hole_vertices)/n
        return self._ray_march(hole_pt, hole_vertices)

    @staticmethod
    def _ray_march(target, hole_vertices):
        crossed_left_segments = []
        crossed_right_segments = []

        previous_x, previous_y = hole_vertices[-1]
        target_x, target_y = target
        for current_x, current_y in hole_vertices:
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

    @staticmethod
    def _compute_winding(pts):
        previous_pt = pts[-1]
        sum = 0
        for p in pts:
            sum += previous_pt[0]*p[1] - previous_pt[1]*p[0]
            previous_pt = p
        return np.sign(sum)

    def face(self, item):
        return self._face_vertices[item]

    def side(self, item):
        return self._side_perimeter[item]
