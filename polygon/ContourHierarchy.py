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