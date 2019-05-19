from typing import Iterator, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from polygon.polygon import VerticesList
from polygon.transform import step_to_slope, merge_slope

SCALE = 6
WORLD_DEPTH = 7*SCALE


class Viewer(object):
    def __init__(self, file_name):
        # self._image = cv2.imread(file_name)
        image = cv2.imread(file_name)
        height, width, _ = image.shape
        image[height - 1, 0] = (0, 0, 0)
        # self._image = image
        image = cv2.resize(image, (width*SCALE, height*SCALE), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray

        # thickening
        # must be so that WITH A TRANSLATION > 1 THEN THE SIDE OF A PIXEL IS NOT ON SCALE BUT THE SIDE OF A SLOPE IS
        g1 = np.roll(gray, 2, axis=0)
        g2 = np.roll(gray, 2)
        g3 = np.roll(g1, 2)
        g4 = gray + g1 + g2 + g3
        g4[0] = gray[0]
        g4[1] = gray[1]
        g4[:, 0] = gray[:, 0]
        g4[:, 1] = gray[:, 1]
        gray = g4
        self._image = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)

        self.step = []

        # contours is a list of polygons' vertices stored as (x, y)
        self._contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._contour_simplification1()

        shift_corrections = None
        #shift_corrections = self._compute_shift(2, 2)
        #for s, cor in zip(self._contours, shift_corrections):
        #    cor = np.reshape(cor, (len(cor), 1, 2))
        #    s[:] = s + cor
        self._contour_simplification2(shift_corrections)

        self._ignored = []  # debug stuff

        #self._image = image

    def _contour_simplification1(self):
        """
        simplify inner edge from 2 points to one
        line -> 45° -> 45° -> line becomes line -> 90° -> line
        """
        new_cs = []
        for shape in self._contours:
            new_c = VerticesList()
            prev_p = shape[-1][0]
            n = len(shape)
            i = 0
            while i < n:
                curr_p = shape[i][0]
                dx, dy = curr_p - prev_p
                if abs(dx) == abs(dy) == 1:
                    new_p = prev_p + [dx, 0]
                    if self.get_at(*new_p):
                        new_c.replace_last(new_p)
                    else:
                        new_p = prev_p + [0, dy]
                        if self.get_at(*new_p):
                            new_c.replace_last(new_p)
                        else:
                            new_c.append(curr_p)
                else:
                    new_c.append(curr_p)

                prev_p = new_c[-1]
                i += 1
            new_cs.append(np.array(new_c))
            new_cs[-1].resize((len(new_c), 1, 2))
        self._contours = new_cs

    def _contour_simplification2(self, shift_correction=None):
        new_cs = []
        if shift_correction is None:
            shift_correction = [None]*len(self._contours)
        for shape, correction in zip(self._contours, shift_correction):
            new_cs.append(merge_slope(step_to_slope(shape, correction)))
        self._contours = new_cs

    def get_at(self, x, y):
        """
        :param x: position on mask
        :param y: positon on mask
        :return: true if mask is solid at position
        :rtype: bool
        """
        try:
            return not np.all(self._image[y, x] == (255, 255, 255))
        except IndexError:
            return True

    def _compute_shift(self, dx: int, dy: int):
        correction_collection = [np.zeros(shape=(len(_), 2), dtype=np.int) for _ in self._contours]
        for shape, correction in zip(self._contours, correction_collection):
            for p, c in zip(shape, correction):
                p = p[0]
                if self.get_at(*(p + [-1, -1])):
                    if self.get_at(*(p + [1, 1])):
                        if self.get_at(*(p + [-1, 1])):
                            c[0] = -dx
                        elif self.get_at(*(p + [1, -1])):
                            c[1] = -dy
                    else:
                        c[:] = [-dx, -dy]
                else:
                    if self.get_at(*(p + [1, 1])):
                        pass
                    elif self.get_at(*(p + [-1, 1])):
                        c[0] = -dx
                    elif self.get_at(*(p + [1, -1])):
                        c[1] = -dy

        return correction_collection

    def show(self, plot=plt.plot()):
        # cv2.drawContours(self._image, self._contours, -1, (192, 166, 64))
        color = self._color_iterator()
        for i, shape in enumerate(self._contours):
            b, g, r = next(color)[0:3]
            plt.plot([], color=(r/255, g/255, b/255), label="{}: {}".format(i, len(shape)))
            for p in shape:
                px, py = p[0]
                self._image[py, px] = (r, g, b)
        plt.imshow(self._image)
        plt.legend(loc="upper left")
        plt.show()
        print("k")

    def _color_iterator(self) -> Iterator[Tuple[int, int, int]]:
        """
        :return: color iterator in 8bits bgr color space
        """
        return iter(
            (int(255*b), int(255*g), int(255*r)) for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(self._contours)))
        )


if __name__ == '__main__':
    m = Viewer("heist.png")
    m.show()
