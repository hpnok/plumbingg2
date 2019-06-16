from collections import OrderedDict
from typing import List, Dict


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

    def get_holes_index(self, parent_index: int) -> List[int]:
        """
        If the parent_index is a hole then this will return a list of polygon
        :param parent_index: Index of the polygon
        :return: list of index of holes in the countour list, direct children of the
        polygon identified by the parent index
        """
        return [_ for _, h in enumerate(self._hierarchy) if h.parent == parent_index]

    def compute_holes_multimap(self) -> Dict[int, List[int]]:
        """
        :return: A mapping between polygon index the the list of their holes' index
        """
        multimap = OrderedDict()
        to_skip = []
        n = len(self._hierarchy)
        for i in range(n):
            if i in to_skip:
                continue
            holes_index = self.get_holes_index(i)
            [to_skip.append(_) for _ in holes_index]
            multimap[i] = holes_index
        return multimap
