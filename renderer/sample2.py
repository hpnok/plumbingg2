from typing import Iterator

import numpy as np
from PIL import Image
from glumpy import gloo, app, glm, gl

from constant import SCALE, WORLD_DEPTH
from gg2.mapparser import GG2Map
from polygon.extractor import ImageToPolygon
from polygon.polygon import Polygon
from renderer.camera import Camera

VERTEX_POSITION = "position"
MODEL_MATRIX = "model"
VIEW_MATRIX = "view"
MAP_TEXTURE = "map_texture"
TEXTURE_MATRIX = "tex_mat"


class VertexBuffer(object):
    def __init__(self, vert: np.ndarray, idx: np.ndarray):
        self._vertices = vert.view(gloo.VertexBuffer)
        if idx.dtype == np.int32:
            idx = idx.astype(np.uint32)
        self._indices = idx.view(gloo.IndexBuffer)
        self._primitive = gl.GL_TRIANGLES

    @staticmethod
    def make_empty_buffer(n: int) -> np.ndarray:
        return np.empty(n, [(VERTEX_POSITION, np.float32, 3)])

    @property
    def primitive(self):
        return self._primitive

    @property
    def buffer(self):
        return self._vertices

    @property
    def indices(self):
        return self._indices


class ExtrudedSurfaceVertexBuffer(VertexBuffer):
    def __init__(self, vert: np.ndarray, triangle_indices: np.ndarray):
        """
        :param vert: array of shape (n, 2)
        :param triangle_indices: triangle indices of shape (m, 3)
        """
        vertices = self.make_empty_buffer(len(vert))
        vertices[VERTEX_POSITION][:, :2] = vert
        vertices[VERTEX_POSITION][:, 2] = 0

        super(ExtrudedSurfaceVertexBuffer, self).__init__(vertices, triangle_indices)


class ExtrudedPerimeterVertexBuffer(VertexBuffer):
    def __init__(self, vert: np.ndarray):
        """
        :param vert: array of shape (n, 2)
        :param contour_indices: contour of shape (m)
        """
        vertices = self.make_empty_buffer(2*len(vert))
        vertices[VERTEX_POSITION][::2, :2] = vert
        vertices[VERTEX_POSITION][::2, 2] = 0
        vertices[VERTEX_POSITION][1::2, :2] = vert
        vertices[VERTEX_POSITION][1::2, 2] = WORLD_DEPTH
        indices = np.array(list(range(2*len(vert))) + [0, 1], dtype=np.uint32)

        super().__init__(vertices, indices)

        self._primitive = gl.GL_TRIANGLE_STRIP


class BackGroundVertexBuffer(VertexBuffer):
    def __init__(self, width: int, height: int):
        vertices = self.make_empty_buffer(4)
        vertices[VERTEX_POSITION] = [
            [0, 0, WORLD_DEPTH],
            [width, 0, WORLD_DEPTH],
            [width, height, WORLD_DEPTH],
            [0, height, WORLD_DEPTH]
        ]
        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        super(BackGroundVertexBuffer, self).__init__(vertices, indices)


class Renderable(object):
    def draw(self, r: gloo.Program, dt: float):
        raise NotImplementedError()


class Level(Renderable):

    def __init__(self, polygons: Iterator[Polygon], width: int, height: int):
        self._buffers = []
        for p in polygons:
            self._buffers.append(ExtrudedSurfaceVertexBuffer(p.face(slice(None)), p.triangles()))
            self._buffers.append(ExtrudedPerimeterVertexBuffer(p.contour.vertices))
            for h in p.holes:
                self._buffers.append(ExtrudedPerimeterVertexBuffer(h.vertices))
        self._model = glm.translation(-width//2, -height//2, 0)
        glm.scale(self._model, 2/max(width, height))
        self._tex_mat = glm.scale(np.eye(4, dtype=np.float32), 1/width, 1/height, 1)

    def draw(self, r: gloo.Program, dt: float):
        r[TEXTURE_MATRIX] = self._tex_mat
        r[MODEL_MATRIX] = self._model
        for b in self._buffers:
            r.bind(b.buffer)
            r.draw(b.primitive, b.indices)


class GG2Level(Level):
    def __init__(self, file_name):
        gg2_map = GG2Map(file_name)
        extractor = ImageToPolygon(gg2_map.mask)
        width, height = gg2_map.width*SCALE, gg2_map.height*SCALE
        super(GG2Level, self).__init__(extractor.get_polygons(), width, height)

        self._bg = BackGroundVertexBuffer(width, height)
        self._tex = self._spaghetto(gg2_map.image, 255 - gg2_map.mask)
        glm.scale(self._tex_mat, 1, 1/2, 1)

    def draw(self, r: gloo.Program, dt: float):
        super(GG2Level, self).draw(r, dt)

        r[TEXTURE_MATRIX] = np.matmul(self._tex_mat, glm.translation(0, 0.5, 0))
        r.bind(self._bg.buffer)
        r.draw(self._bg.primitive, self._bg.indices)

    def _spaghetto(self, raw_image, alpha_mask):
        """
        create a texture where the top part is for front/sides and bottom is for the back

        this thing and the tex matrice shouldn't exist but it's was quicker for prototyping than figure out how to
        swap textures with glumpy, if slopes ever contain normal info it should be used to know what to sample
        """
        back = self._speghetti(raw_image, 255 - alpha_mask)
        front = self._speghetti(raw_image, alpha_mask)
        height, width, depth = front.shape
        output = np.empty((2*height, width, depth), dtype=front.dtype)
        output[:height] = front
        output[height:] = back
        return output

    def _speghetti(self, raw_image: np.ndarray, alpha_mask: np.ndarray):
        """
        :param image: RGBA image
        :param alpha_mask: colorkey mask where 255 is colored and 0 is ignored
        :return:
        """
        raw_image = raw_image.copy()
        # raw_image[:, :, 3] = alpha_mask
        bg_image = Image.fromarray(raw_image)
        original = bg_image.copy()
        mask = Image.fromarray(alpha_mask, 'L')
        bg_image.paste(original, (-1, 0), mask)
        bg_image.paste(original, (1, 0), mask)
        bg_image.paste(original, (0, -1), mask)
        bg_image.paste(original, (0, 1), mask)
        bg_image.paste(original, (0, 0), mask)
        # bg_image.show()
        return np.array(bg_image)


if __name__ == '__main__':
    name = "../gg2/maps/cp_mountainjazz.png"
    level = GG2Level(name)

    window = app.Window(width=960, height=540, color=(0.44, 0.11, 0.73, 1.))

    renderer = gloo.Program(
        vertex='shader/vert.glsl',
        fragment='shader/frag.glsl'
    )
    renderer[MAP_TEXTURE] = level._tex

    camera = Camera()


    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)


    @window.event
    def on_resize(w, h):
        if h != 0:
            ratio = w/float(h)
            renderer['projection'] = glm.perspective(96., ratio, 0.01, 100.)


    @window.event
    def on_draw(dt):
        window.clear()

        camera.update(dt)
        renderer[VIEW_MATRIX] = camera.get_view()

        level.draw(renderer, dt)


    @window.event
    def on_key_press(key_pressed, modifiers):
        camera.set_state(key_pressed, True)


    @window.event
    def on_key_release(key_pressed, modifiers):
        camera.set_state(key_pressed, False)


    app.run()
