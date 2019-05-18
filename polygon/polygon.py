import numpy as np


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
