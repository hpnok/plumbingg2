import numpy as np

from constant import SCALE
from .polygon import OrthogonalVertexList, CyclicList


def step_to_slope(vertices: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """
    :param vertices: alterning orthogonal vectors
    :return:
    """
    v = OrthogonalVertexList(vertices)
    deltas = _compute_deltas(v)

    i, n = 0, v.size
    sloped_polygon = CyclicList(_.copy() for _ in v)
    while i < n:
        delta = deltas[i]
        if abs(delta) == SCALE:
            _apply_sloping(v, deltas, i, sloped_polygon)
        i += 1

    _remove_duplicate(sloped_polygon)

    slope_to_remove = _45_degrees_slope_picker(deltas)

    if correction is not None:
        sloped_polygon = [p+c for p, c in zip(sloped_polygon, correction) if p is not None]

    for start, end in slope_to_remove:
        for i in range(start + 1, end):
            sloped_polygon[i] = None

    return np.array([[_] for _ in sloped_polygon if _ is not None])  # (n, 1, 2) shape


def _apply_sloping(vertices: OrthogonalVertexList, deltas: CyclicList, current_index: int, sloped_polygon: CyclicList):
    previous_delta = deltas[current_index - 1]
    next_delta = deltas[current_index + 1]
    previous_direction = vertices[current_index - 1] - vertices[current_index - 2]
    next_direction = vertices[current_index + 1] - vertices[current_index]
    if abs(previous_delta - next_delta) <= SCALE:
        sloped_polygon[current_index - 1] -= previous_direction//2
        sloped_polygon[current_index] += next_direction//2
    else:
        d = min(abs(previous_delta), abs(next_delta))//2
        sloped_polygon[current_index - 1] -= (previous_direction*d)//abs(previous_delta)
        sloped_polygon[current_index] += (next_direction*d)//abs(next_delta)


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


def _segment_to_slope(vertices: OrthogonalVertexList, deltas: CyclicList, first_index: int, breaking_index: int, sloped_polygon: list):
    """
    Add pts including breaking_index in sloped_polygon
    :param vertices: points of a polygon
    :param deltas: list of delta
    :param first_index: last index setted, previous point in the first step
    :param breaking_index: index on which the slop ends
    :param sloped_polygon: polygon being built
    :return:
    """
    # TODO: handle all as a single case?
    # TODO: handle slope that change direction as long as it keeps SCALE sized deltas
    step_direction = deltas[first_index + 1]
    breaking_step_direction = deltas[breaking_index]
    end_in_inverse_direction = step_direction*breaking_step_direction < 0
    if breaking_index - first_index == 3:
        if end_in_inverse_direction:
            shift = 0.5*SCALE*np.sign(vertices[first_index] - vertices[first_index - 1])
            if sloped_polygon[first_index] is None:
                sloped_polygon[first_index] = vertices[first_index]
            sloped_polygon[first_index] -= shift
            if abs(breaking_step_direction) == SCALE:
                sloped_polygon[first_index + 1] = (vertices[first_index + 1] + vertices[first_index + 2])/2
                sloped_polygon[first_index + 2] = None
                sloped_polygon[breaking_index] = vertices[breaking_index] + shift
            else:
                sloped_polygon[first_index + 1] = vertices[first_index + 1] + shift
                sloped_polygon[first_index + 2] = vertices[first_index + 2]
                sloped_polygon[breaking_index] = vertices[breaking_index]
        else:
            sloped_polygon[first_index + 1] = None
            sloped_polygon[first_index + 2] = vertices[first_index + 2]
            sloped_polygon[breaking_index] = vertices[breaking_index]

    elif breaking_index - first_index == 5:
        # if diff ± SCALE merge
        pass
    elif breaking_index - first_index > 6:
        # if diff ± SCALE merge
        pass
    else:
        raise ValueError("indexes received {}, {}".format(first_index, breaking_index))


def _compute_deltas(v: OrthogonalVertexList) -> CyclicList:
    deltas = CyclicList()
    prev = v[-1]
    for curr in v:
        dx, dy = curr - prev
        deltas.append(dx + dy)
        prev = curr

    return deltas
