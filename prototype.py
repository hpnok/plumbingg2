import io

import cv2
import numpy as np
from matplotlib import pyplot as plt
import triangle as tr
from typing import *

from intersection import line_with_line

"""
NOTE: there's somekind of randomization in the triangle lib that fail on edge case of a floating diagonal line but pass sometimes
    TODO: fix it while simplifying contour, do 8 patterns?

TODO: load mask

transform bit mask to poly

TODO: simplify contour

transform poly to triangles

#TODO: make sure only sides (no mid triangles) are rendered while creating side triangles (poly with hole)
#TODO: work in 1x and handle 1 pixel line as a special case (would allow slopes!!)

using contour hierarchy handle polygons with hole
    find point inside hole
"""

SCALE = 6
WORLD_DEPTH = 7*SCALE


class ContourLevel(object):
    def __init__(self, next_on_same_level, previous_on_same_level, first_child, parent):
        super(ContourLevel, self).__init__()
        self.next_on_same_level = next_on_same_level
        self.previous_on_same_level = previous_on_same_level
        self.first_child = first_child
        self.parent = parent


class ContourHierarchy(object):
    def __init__(self, hierarchy):
        self._hierarchy = [ContourLevel(*_) for _ in hierarchy[0]]

    def get_holes_index(self, parent_index):
        return [_ for _, h in enumerate(self._hierarchy) if h.parent == parent_index]


class STLWriter(object):
    def __init__(self):
        self._tri_count = 0
        self._stl = io.StringIO()

    def push(self, tri_pts: Tuple[Tuple[float, float, float]] or List[List[float, float, float]]
             , normal: Tuple[float, float, float]):
        self._tri_count += 1
        nx, ny, nz = normal
        v1x, v1y, v1z = tri_pts[0]
        v2x, v2y, v2z = tri_pts[1]
        v3x, v3y, v3z = tri_pts[2]
        self._stl.write("""
    facet normal {nx:e} {ny:e} {nz:e}
        outer loop
            vertex {v1x:e} {v1y:e} {v1z:e}
            vertex {v2x:e} {v2y:e} {v2z:e}
            vertex {v3x:e} {v3y:e} {v3z:e}
        endloop
    endfacet\n""".format(
            nx=nx, ny=ny, nz=nz,
            v1x=v1x, v1y=v1y, v1z=v1z,
            v2x=v2x, v2y=v2y, v2z=v2z,
            v3x=v3x, v3y=v3y, v3z=v3z))

    def save(self, file_name):
        if self._tri_count == 0:
            print("No triangle pushed")
            return
        with open(file_name, "w") as f:
            f.write("solid gg2")
            f.write(self._stl.getvalue())
            f.write("endsolid gg2")


class Polygon(object):
    def __init__(self, pts: np.ndarray, holes: List[np.ndarray] = None):
        n = len(pts)
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
        self._contour = pts
        self._winding = self._compute_winding(pts)

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
        ray_start = np.copy(hole_pt)
        ray_start[0] = min(hole_vertices[:, 0]) - 1
        crossed, hole_pt = self._ray_march(ray_start, hole_pt, hole_vertices)
        if crossed is None:  # if it doesn't intersects from the left then it most from the right
            ray_start[0] = max(hole_vertices[:, 0]) + 1
            crossed, hole_pt = self._ray_march(ray_start, hole_pt, hole_vertices)
            if crossed is None:
                raise RuntimeError("Could't find a point inside the hole, write some unit test :^)")
        return hole_pt

    @staticmethod
    def _ray_march(start, target, hole_vertices):
        # FIXME: segments order isn't guaranteed
        """
        keep 2 lists intersecting edge index with the x intersection position
        one for the right side one for the left
        if centroid isn't inside, sort and pick average of last 2
        """
        crossed = None
        previous_pt = hole_vertices[-1]
        for pt in hole_vertices:
            r = line_with_line(previous_pt, pt, start, target)
            if r is None:
                continue
            dy = pt[1] - previous_pt[1]
            if (dy > 0 and np.all(r != previous_pt)) or (dy < 0 and np.all(r != pt)):
                if crossed is None:
                    crossed = r
                else:
                    target = (r + crossed)/2
                    break
        return crossed, target

    @staticmethod
    def _compute_winding(pts):
        previous_pt = pts[-1]
        sum = 0
        for p in pts:
            sum += previous_pt[0]*p[1] - previous_pt[1]*p[0]
            previous_pt = p
        return np.sign(sum)

    def write_to(self, stl_writer):
        self._write_face_triangle(stl_writer)
        self._write_side_triangles(stl_writer)

    def _write_side_triangles(self, stl_writer):
        w = (0, 0, self._winding)
        previous_pt = self._contour[-1]
        for pt in self._contour:
            v = pt - previous_pt
            n = np.cross(v, w)
            n = n/np.linalg.norm(n)
            x1, y1 = previous_pt
            x2, y2 = pt
            t1 = ((x1, y1, 0), (x2, y2, 0), (x1, y1, -WORLD_DEPTH))
            stl_writer.push(t1, n)
            t2 = ((x1, y1, -WORLD_DEPTH), (x2, y2, 0), (x2, y2, -WORLD_DEPTH))
            stl_writer.push(t2, n)
            previous_pt = pt

    def _write_face_triangle(self, stl_writer: STLWriter):
        n = (0, 0, 1)
        pts = self._contour
        for tri in self._plane_triangles:
            tri_pts = [list(pts[_]) + [0] for _ in tri]
            stl_writer.push(tri_pts, n)


class Mask(object):
    def __init__(self, file_name):
        # self._image = cv2.imread(file_name)
        image = cv2.imread(file_name)
        height, width, _ = image.shape
        image[height - 1, 0] = (0, 0, 0)
        self._image = cv2.resize(image, (width*SCALE, height*SCALE), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        # contours is a list of polygons' vertices stored as (x, y)
        self._contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._hierarchy = ContourHierarchy(hierarchy)

        self._ignored = []  # debug stuff

    def get_polygons(self):
        to_skip = []
        n = len(self._contours)
        for i in range(n):
            if i in to_skip:
                continue
            holes_index = self._hierarchy.get_holes_index(i)
            [to_skip.append(_) for _ in holes_index]
            poly = self._contours[i]
            holes = [self._contours[_] for _ in holes_index]
            print(i, " start")
            yield Polygon(poly, holes)
            print(i, " ok")
        self._ignored = [self._contours[_] for _ in to_skip]

    def show_scatter(self):
        plt.imshow(self._image, cmap='gray')
        i = -1
        for l in self._contours:
            i += 1
            x, y = np.transpose(l)
            plt.scatter(x, y, label=str(i) + ". ")
        plt.legend(loc="upper left")
        plt.show()

    def show(self):
        # cv2.drawContours(self._image, self._contours, -1, (192, 166, 64))
        color = self._color_iterator()
        triangle = iter(self._triangulate())
        for shape in self._contours:
            c = next(color)[0:3]
            t = next(triangle)
            for s in self._ignored:
                if s is shape:
                    break
            else:
                self._draw_poly_tris(shape, t, c)
            """for pts in shape:
                p = pts[0]
                self._image[p[1], p[0]] = c"""
        plt.imshow(self._image)
        plt.show()

    def _triangulate(self) -> List[np.ndarray]:
        polygons = list()
        for polygon in self._contours:
            n = len(polygon)
            p = np.resize(polygon, (n, 2))
            s = np.array([(i, (i + 1)%n) for i in range(n)])
            poly_dict = dict(vertices=p, segments=s)
            tri_dict = tr.triangulate(poly_dict, 'p')
            polygons.append(tri_dict["triangles"])
        return polygons

    def _draw_poly_tris(self, pts: np.ndarray, tris: np.ndarray, color: Tuple[int, int, int]):
        pts = np.resize(pts, (len(pts), 2))
        for tri in tris:
            # compute centroid get pixel on map, if black then draw
            tri_pts = np.array([pts[_] for _ in tri], dtype=np.int32)
            # cx, cy = [sum(_) for _ in tri_pts.T]
            # if all(self._image[cy//3, cx//3] == self.SOLID_BGR):
            cv2.polylines(self._image, [tri_pts], 1, color)

    @staticmethod
    def _inverse(grayscale_image: np.ndarray) -> np.ndarray:
        return 255 - grayscale_image

    def _color_iterator(self) -> Iterator[Tuple[int, int, int]]:
        """
        :return: color iterator in 8bits bgr color space
        """
        return iter(
            (int(255*b), int(255*g), int(255*r)) for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(self._contours)))
        )


if __name__ == "__main__":
    m = Mask("heist.png")
    stl = STLWriter()
    #for p in m.get_polygons():
    #    p.write_to(stl)
    stl.save("heist.stl")
    print(m._hierarchy)
    m.show_scatter()
    #m.show()
