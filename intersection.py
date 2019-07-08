from typing import List

import numpy as np


def line_with_line(p0, p1, q0, q1):
    """
    http://geomalgorithms.com/a05-_intersect-1.html
    """
    u = p1 - p0
    v = q1 - q0
    v_norm = np.array([v[1], -v[0]])
    det = np.dot(v_norm, u)
    if det == 0:  # parallel or (u and or v are 0)
        return None
    w = q0 - p0
    s = np.dot(v_norm, w)/det
    if s < 0 or s > 1:
        return None
    u_norm = np.array([u[1], -u[0]])
    t = np.dot(u_norm, w)/det  # minus sign canceled since u_norm dot v = - v_norm dot u, using the same rotation to get the normal vector
    if t < 0 or t > 1:
        return None
    return p0 + s*u


def line_with_plane(p0: np.ndarray,
                    p1: np.ndarray,
                    n_plane: np.ndarray,
                    p_plane: np.ndarray) -> np.ndarray or None:
    det = np.dot(n_plane, p1 - p0)
    if det == 0:
        return None
    k: float = np.dot(n_plane, p_plane - p0)/det
    return p0 + k*(p1 - p0)


def pnpoly(vertices, test_pt):
    """
    https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    """
    test_pt_y_is_within_edge = lambda y1, y2: (y1 > test_pt.y) != (y2 > test_pt.y)
    previous_pt = vertices[-1]
    is_inside = False  # at infinite can't be inside a polygon which is finite
    for current_pt in vertices:
        if test_pt_y_is_within_edge(current_pt.y, previous_pt.y):
            x_value_on_edge_at_y = lambda y: ((current_pt.x - previous_pt.x)/(current_pt.y - previous_pt.y))*(y - previous_pt.y) + previous_pt.x
            x_intersection = x_value_on_edge_at_y(test_pt.y)
            if x_intersection < test_pt.x:  # assumes comes from infinite left
                is_inside = not is_inside
