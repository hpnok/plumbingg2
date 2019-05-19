import cv2
import numpy as np

from constant import SCALE
from polygon.ContourHierarchy import ContourHierarchy
from polygon.polygon import Polygon
from polygon.transform import to_orthogonal_contour, step_to_slope, merge_slope


def get_gg2_image(file_name: str):
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    image[height - 1, 0] = 0
    return image


class ImageToPolygon(object):
    def __init__(self, grayscale_image):
        grayscale_image = self._to_binary(grayscale_image)
        inverted_grayscale_image = self._invert(grayscale_image)
        inverted_grayscale_image = self._expend(inverted_grayscale_image, 2, 2)
        self._contours, hierarchy = cv2.findContours(inverted_grayscale_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._hierarchy = ContourHierarchy(hierarchy)

        self._image = self._invert(inverted_grayscale_image)

        self._simplify_contours()

    def get_polygons(self):
        to_skip = []
        n = len(self._contours)
        for i in range(n):
            if i in to_skip:
                continue
            if i == 18:
                print("18")
            holes_index = self._hierarchy.get_holes_index(i)
            [to_skip.append(_) for _ in holes_index]
            poly = self._contours[i]
            holes = [self._contours[_] for _ in holes_index]
            print(i, " start")
            yield Polygon(poly, holes)
            print(i, " ok")

    def _simplify_contours(self):
        new_contours = []
        for shape in self._contours:
            orthogonal_contour = to_orthogonal_contour(shape, self._image)
            contour = step_to_slope(orthogonal_contour)
            contour = merge_slope(contour)
            new_contours.append(contour)
        self._contours = new_contours

    def _expend(self, image, dx, dy):
        """
        By diagonally smearing the image it get expended and pixels
        connected diagonally become part of the same polygon
        With dx and dy > 1 then the side of a pixel is not equal to SCALE but the side of a slope step is
        :param image: image to expend
        :param dx: horizontal shift
        :param dy: vertical shift
        :return:
        """

        assert dx > 0 and dy > 0
        height, width = image.shape
        image = cv2.resize(image, (width*SCALE, height*SCALE), interpolation=cv2.INTER_NEAREST)

        horizontal_edge = image[:dy].copy()
        vertical_edge = image[:, :dx].copy()
        image += np.roll(image, dx, axis=0)
        image += np.roll(image, dy)
        image[:dy] = horizontal_edge
        image[:, :dx] = vertical_edge

        return self._to_binary(image)

    @staticmethod
    def _invert(image):
        return 255 - image

    @staticmethod
    def _to_binary(image):
        """
        All non zero pixel of image are set to 255
        :param image: input image
        :return: image transformed
        """
        return 255*np.clip(image, None, 1)
