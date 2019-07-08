import numpy as np
import pygame as pg
from glumpy import app, gloo, gl, glm

from constant import WORLD_DEPTH
from gg2.mapparser import GG2Map
from gg2.sample import MapCollider, CharacterClass, call_action
from gg2 import sample
from intersection import line_with_plane
from renderer.camera import Camera
from renderer.sample2 import GG2Level, MAP_TEXTURE, VIEW_MATRIX, MODEL_MATRIX, VERTEX_POSITION, VertexBuffer, PROJECTION_MATRIX


def _set_player_vertices(r: gloo.Program, player_rect: pg.Rect, depth: float):
    w, h = player_rect.size
    player_vertex = VertexBuffer.make_empty_buffer(4)
    player_vertex[VERTEX_POSITION] = [
        [0, 0, depth],
        [w, 0, depth],
        [w, h, depth],
        [0, h, depth],
    ]
    _player_plane = VertexBuffer(player_vertex, np.array([[0, 1, 2], [0, 2, 3]]))
    r[VERTEX_POSITION] = _player_plane._vertices
    return _player_plane


def _render_player(r: gloo.Program, player_rect: pg.Rect):
    #model_matrix = glm.translation(player_rect.x, player_rect.y, 0)
    r[MODEL_MATRIX] = glm.translation(player_rect.x, player_rect.y, 0)
    #r[MODEL_MATRIX] = np.matmul(model_matrix, level._model)
    r.draw(mode=gl.GL_TRIANGLES, indices=player_plane.indices)


def _track_player(c: Camera, _player: CharacterClass):
    cx, cy = _player._rect.midtop
    #cx += _player.vxs
    #cy += _player.vys
    c.set_position(x=cx, y=cy)


if __name__ == '__main__':
    name = "gg2/maps/koth_gallery.png"

    current_map = GG2Map(name)
    level = GG2Level(gg2_map=current_map)
    map_collider = MapCollider(current_map)

    player = CharacterClass(pg.Rect(1600, 800, 16, 36), pg.Mask((16, 36)), (0, 0), map_collider)
    player.mask.fill()
    player._jump_speed = -318
    player.max_x_speed = 300
    player.max_y_speed = 800

    sample.key_mapping = {
        ord('A'): CharacterClass.Action.MOVE_LEFT,
        ord('D'): CharacterClass.Action.MOVE_RIGHT,
        ord('W'): CharacterClass.Action.JUMP}

    window = app.Window(width=960, height=540, color=(0.44, 0.11, 0.73, 1.))

    renderer = gloo.Program(
        vertex='renderer/shader/vert.glsl',
        fragment='renderer/shader/frag.glsl'
    )

    player_renderer = gloo.Program(
        vertex='renderer/shader/player_vert.glsl',
        fragment='renderer/shader/player_frag.glsl'
    )
    player_plane = _set_player_vertices(player_renderer, player._rect, 6)

    renderer[MAP_TEXTURE] = level._tex

    camera = Camera(z=-180)


    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)


    @window.event
    def on_resize(w, h):
        if h != 0:
            ratio = w/float(h)
            renderer[PROJECTION_MATRIX] = glm.perspective(96., ratio, 0.01, 1200.)
            player_renderer[PROJECTION_MATRIX] = renderer[PROJECTION_MATRIX]


    @window.event
    def on_draw(dt):
        window.clear()

        #camera.update(dt)
        player.update(dt)
        _track_player(camera, player)

        renderer[VIEW_MATRIX] = camera.get_view()

        level.draw(renderer, dt)

        player_renderer[VIEW_MATRIX] = camera.get_view()
        _render_player(player_renderer, player._rect)


    @window.event
    def on_key_press(key_pressed, modifiers):
        # camera.set_state(key_pressed, True)
        call_action(player, key_pressed, True)


    @window.event
    def on_key_release(key_pressed, modifiers):
        # camera.set_state(key_pressed, False)
        call_action(player, key_pressed, False)


    @window.event
    def on_mouse_press(x, y, button):
        if button == 1:
            p0, p1 = camera.to_world_coordinate([x, y])
            p = line_with_plane(p0[:3], p1[:3], np.array([0, 0, 1], dtype=np.float32), np.array([0, 0, 0], dtype=np.float32))
            player._rect.centerx = (int(p[0]), int(p[1]))


    app.run()
