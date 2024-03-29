from typing import List, Tuple, Set

import numpy as np

from constant import SCALE
from polygon.contourhierarchy import ContourHierarchy
from .polygon import OrthogonalVertexList, CyclicList, VerticesList


def to_orthogonal_contour(vertices: np.ndarray, binary_image: np.ndarray) -> np.ndarray:
    """
    2 point neighbor diagonaly get merged as a single one that is still inside the map
    :param vertices: of a polygon
    :param binary_image: image where 0 are solids
    :return:
    """
    get_at = lambda p: binary_image[p[1], p[0]] == 0
    v = OrthogonalVertexList(vertices)
    new_contour = VerticesList()

    previous_point = v[-1]
    i, n = 0, v.size
    while i < n:
        current_point = v[i]
        dx, dy = current_point - previous_point
        if abs(dx) == abs(dy) == 1:
            test_point = previous_point + [dx, 0]
            if get_at(test_point):
                new_contour.replace_last(test_point)
            else:
                test_point = previous_point + [0, dy]
                if get_at(test_point):
                    new_contour.replace_last(test_point)
                else:
                    new_contour.append(current_point)
        else:
            new_contour.append(current_point)

        previous_point = new_contour.last_vertex_added()
        i += 1

    return np.reshape(np.array(new_contour), (len(new_contour), 1, 2))


def step_to_slope(vertices: np.ndarray, correction: np.ndarray = None) -> np.ndarray:
    """
    extrapolate slope from a list of edges which are all at 90° of each other
    :param vertices: alterning orthogonal vectors
    :param correction: list of vertices correction to apply (optional)
    :return:
    """
    v = OrthogonalVertexList(vertices)
    deltas = _compute_deltas(v)

    i, n = 0, v.size
    sloped_polygon = CyclicList(_.astype(np.float) for _ in v)
    while i < n:
        delta = deltas[i]
        if abs(delta) == SCALE:
            _apply_sloping(v, deltas, i, sloped_polygon)
        i += 1

    if correction is not None:
        _remove_duplicate_with_correction(sloped_polygon, correction)
    else:
        _remove_duplicate(sloped_polygon)

    slope_to_remove = _45_degrees_slope_picker(deltas)

    if correction is not None:
        sloped_polygon = [p + c for p, c in zip(sloped_polygon, correction) if p is not None]

    for start, end in slope_to_remove:
        for i in range(start + 1, end):
            sloped_polygon[i] = None

    return np.array([[_] for _ in sloped_polygon if _ is not None])  # (n, 1, 2) shape


def merge_slope(vertices: np.ndarray):
    """
    if two consecutive edges have the same angle, they are merged
    :param vertices:
    :return:
    """
    diagonals = CyclicList()
    v = CyclicList(vertices)
    prev = v[-1]
    for curr in v:
        diagonals.append(curr - prev)
        prev = curr
    assert all(np.any(_ != [0, 0]) for _ in diagonals)
    is_diagonal = lambda p: np.all(p != 0)
    diagonals = [_[0] if is_diagonal(_) else None for _ in diagonals]

    is_colinear = lambda u, w: u[0]*w[1] == u[1]*w[0]
    i, n = 0, len(v)
    while (diagonals[i] is not None and diagonals[i - 1] is not None
           and is_colinear(diagonals[i], diagonals[i - 1]) and n > 0):
        i -= 1
        n -= 1
    while i < n:
        if diagonals[i] is not None and diagonals[i - 1] is not None:
            if is_colinear(diagonals[i], diagonals[i - 1]):
                v[i - 1] = None
                diagonals[i] += diagonals[i - 1]
        i += 1
    return np.array([_ for _ in v if _ is not None])


def correction(vertices: np.ndarray, dx, dy):
    """ INPLACE
    TODO: don't make it inplace just like the other transforms?
    undo the shift that was applied to the image to connect pixels of a slope
    use normal of the neighbors of a vertex to determine if it must be moved"""
    previous_p = vertices[-1][0]
    sum = 0
    for p in vertices:
        p = p[0]
        sum += previous_p[0]*p[1] - previous_p[1]*p[0]
        previous_p = p
    #  w = (0, 0, np.sign(sum))  # winding direction
    w = (0, 0, -1)  # winding direction

    correction_per_segment = []
    previous_p = vertices[-1][0]
    for p in vertices:
        p = p[0]
        v = p - previous_p
        n = np.cross(v, w)
        n = n/np.linalg.norm(n)
        x = dx if n[0] > 0 else 0
        y = dy if n[1] > 0 else 0
        correction_per_segment.append((x, y))
        previous_p = p

    i = -1
    n = len(vertices)
    while i < n - 1:
        dx1, dy1 = correction_per_segment[i]
        dx2, dy2 = correction_per_segment[i + 1]
        vertices[i] -= [max(dx1, dx2), max(dy1, dy2)]
        i += 1


def merge_holes(polygon_list: np.ndarray, holes_index: List[int]):
    # a vertex has 2 edges the new point is the intersection of the orthogonal edges of the pair of points found.
    # 1: detect hole that must be connected togheter and mark the indexes
    connection_mapping = {}
    for i in holes_index[:-1]:
        hole_i = polygon_list[i]
        for j in holes_index[i + 1:]:
            hole_j = polygon_list[j]
            connecting_pairs = _connect_hole(hole_i, hole_j)
            if connecting_pairs:
                connection_mapping[(i, j)] = connecting_pairs
    hole_union: List[Set[int]] = []
    for pair in connection_mapping:
        pair_set = set(pair)
        for union in hole_union:
            if union.intersection(pair_set):
                union |= pair_set
                break
        else:
            hole_union.append(pair_set)
    vertex_mapping = {index: [None]*len(polygon_list[index]) for index in holes_index}
    for union in hole_union:
        for hole_index in sorted(union):
            new_poly = []
            #  TODO: merge holes into a singe one, update contour hierarchy
    # hole_union is the list of new holes, each set containt
    return np.array([])


def _connect_hole(hole1: np.ndarray, hole2: np.ndarray) -> List[Tuple[int, int]]:
    """
    :param hole1:
    :param hole2:
    :return: list of connected vertex index of the two hole
    """
    connecting_pairs = []
    for i in range(len(hole1)):
        pi = hole1[i][0]
        for j in range(len(hole2)):
            pj = hole2[j][0]
            dx, dy = pj - pi
            if 1 == abs(dx) == abs(dy):
                connecting_pairs.append((i, j))
    return connecting_pairs


def _apply_sloping(vertices: OrthogonalVertexList, deltas: CyclicList, current_index: int, sloped_polygon: CyclicList):
    previous_delta = deltas[current_index - 1]
    next_delta = deltas[current_index + 1]
    previous_direction = vertices[current_index - 1] - vertices[current_index - 2]
    next_direction = vertices[current_index + 1] - vertices[current_index]
    if abs(previous_delta - next_delta) <= SCALE:
        sloped_polygon[current_index - 1] -= previous_direction/2
        sloped_polygon[current_index] += next_direction/2
    else:
        d = min(abs(previous_delta), abs(next_delta))/2
        sloped_polygon[current_index - 1] -= (previous_direction*d)/abs(previous_delta)
        sloped_polygon[current_index] += (next_direction*d)/abs(next_delta)


def _45_degrees_slope_picker(deltas: CyclicList) -> list:
    segment_index = []
    n = len(deltas)
    i = 0
    while abs(deltas[i]) == SCALE:  # if we start in a diagonal, ignore it
        i += 1
    while i < n:
        if abs(deltas[i]) == SCALE:
            slope_start_candidate = i - 1
            while abs(deltas[i]) == SCALE:
                i += 1
            slope_end_candidate = i - 1
            if slope_end_candidate - slope_start_candidate >= 3:
                segment_index.append((slope_start_candidate, slope_end_candidate))
        i += 1

    return segment_index


def _remove_duplicate(sloped_polygon: CyclicList):
    for i in range(len(sloped_polygon)):
        if sloped_polygon[i - 1] is not None and np.all(sloped_polygon[i] == sloped_polygon[i - 1]):
            sloped_polygon[i] = None


def _remove_duplicate_with_correction(sloped_polygon: CyclicList, correction: np.ndarray):
    for i in range(len(sloped_polygon)):
        if sloped_polygon[i - 1] is not None and abs(sum(sloped_polygon[i] - sloped_polygon[i - 1])) <= 1:
            sloped_polygon[i - 1] = (sloped_polygon[i] + sloped_polygon[i - 1] + correction[i] + correction[i - 1])//2
            sloped_polygon[i] = None


def _compute_deltas(v: OrthogonalVertexList) -> CyclicList:
    deltas = CyclicList()
    prev = v[-1]
    for curr in v:
        dx, dy = curr - prev
        deltas.append(dx + dy)
        prev = curr

    return deltas
