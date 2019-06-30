import io
from typing import Tuple, List

import numpy as np

from constant import WORLD_DEPTH
from polygon.polygon import Polygon


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

    def write(self, polyon: Polygon):
        self._write_face_triangle(polyon)
        self._write_side_triangles(polyon)

    def _write_side_triangles(self, polygon: Polygon):
        w = (0, 0, polygon.winding)
        for contour in [polygon.contour] + polygon.holes:
            for i in range(len(contour)):
                v1, v2 = i, (i + 1)%len(contour)
                previous, current = contour[v1], contour[v2]
                v = current - previous
                n = np.cross(v, w)
                n = n/np.linalg.norm(n)
                x1, y1 = previous
                x2, y2 = current
                t1 = ((x1, y1, 0), (x2, y2, 0), (x1, y1, -WORLD_DEPTH))
                self.push(t1, n)
                t2 = ((x1, y1, -WORLD_DEPTH), (x2, y2, 0), (x2, y2, -WORLD_DEPTH))
                self.push(t2, n)

    def _write_face_triangle(self, polygon: Polygon):
        n = (0, 0, 1)
        for tri in polygon._plane_triangles:
            tri_pts = [list(polygon.face(_)) + [0] for _ in tri]
            self.push(tri_pts, n)
