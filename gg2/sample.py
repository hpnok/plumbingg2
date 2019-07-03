# simple implementation of gg2 engine in pygame
# start the engine in another process and pipe the objects coordinates?
import time
from enum import Enum, auto
from typing import Tuple, Callable

import pygame as pg
from math import trunc
from numpy import sign

from constant import SCALE
from gg2.mapparser import GG2Map

BUFFERED_JUMP_GRACE = 0.1
FRICTION = 9.7138714992377222758124853299492e-15  # from py gg2 which took it from gml gg2?
GRAVITY = 700


class DynamicObject(object):
    """An objet which may change in time"""

    def __init__(self, **kwargs):
        super(DynamicObject, self).__init__(**kwargs)
        self.life_time = 0

    def update(self, dt: float, **kwargs):
        self.life_time += dt


class MapCollider(object):
    def __init__(self, gg2_map: GG2Map):
        mask_as_surface = pg.surfarray.make_surface(255 - gg2_map.mask.swapaxes(0, 1))  # invert
        mask_as_surface = pg.transform.scale(mask_as_surface, (SCALE*mask_as_surface.get_width(), SCALE*mask_as_surface.get_height()))
        mask_as_surface.set_colorkey((0, 0, 0))
        self._mask = pg.mask.from_surface(mask_as_surface)

    def overlap(self, other_mask: pg.Mask, offset: Tuple[int, int]):
        # do a fast check using the low res mask first?
        # should wrap mask overlap to return true on out of bound collision?
        return self._mask.overlap(other_mask, offset)


class InputDispatcherMixin(object):
    def __init__(self, input_mapping: dict = None, **kwargs):
        """
        :param input_mapping: A map associating an action identifier (enum, string, pyobject, ...), to a setter
        :return:
        """
        super(InputDispatcherMixin, self).__init__(**kwargs)
        self._input_mapping = input_mapping if input_mapping is not None else dict()

    def register_action(self, action_name, method: Callable):
        """
        example:
        self.register_action("JUMP", self.jump)
        """
        self._input_mapping[action_name] = method

    def dispatch_input(self, action_new_value: dict):
        for action, value in action_new_value.items():
            self._input_mapping[action](value)


class MaskPlatformingMixin(object):
    def __init__(self, rect: pg.Rect = None, mask: pg.Mask = None, mask_offset: Tuple[int, int] = None, jump_speed: float = None, **kwargs):
        """
        :param rect: rect of the collider
        :param mask: shape of the collider
        :param mask_offset: position of the collider from the rect
        """
        super(MaskPlatformingMixin, self).__init__(**kwargs)
        self._jump_buffer = 0
        self.on_ground = False
        self._rect = rect
        self.mask = mask
        self._mox, self._moy = mask_offset
        self._jump_speed = jump_speed

        self.vel_x, self.vel_y = 0, 0
        self.vxs, self.vys = 0, 0

        self.life_time = 0

    def ground_check(self, world_collider: MapCollider):
        if self.vel_y >= 0:
            x = self._rect.x + self._mox
            y = self._rect.y + self._moy
            if not self._is_on_ground_at(x, y, world_collider):
                self.fall()

    def set_jump(self, input_value: bool):
        if not input_value:
            self._jump_buffer = 0
            return
        self._jump_buffer = self.life_time
        self.jump()

    def jump(self):
        if self.on_ground:
            self.vel_y = self._jump_speed
            self.on_ground = False

    def fall(self):
        self.on_ground = False

    def land(self):
        self.on_ground = True
        if self._jump_buffer and self.life_time - self._jump_buffer < BUFFERED_JUMP_GRACE:
            self.jump()

    def move_x(self, x: float):
        self._rect.x = x

    def move_y(self, y: float):
        self._rect.y = y

    def _is_on_ground_at(self, x: int, y: int, world_collider: MapCollider):
        return world_collider.overlap(self.mask, (x, y + 1))

    def _compute_deltas(self, dt: float) -> Tuple[int, int]:
        step_x, step_y = dt*self.vel_x, dt*self.vel_y
        dx, dy = trunc(step_x + self.vxs), trunc(step_y + self.vys)
        self.vxs += step_x - dx
        self.vys += step_y - dy
        return dx, dy

    def pixel_platforming(self, dx: int, dy: int, world_collider: MapCollider) -> Tuple[int, int]:
        """
        quick reimplementation of gg2 platforming
        :param dx: attempted integer displacement in x
        :param dy: attempted integer displacement in y
        :param world_collider: object on which to test collision
        :return:
        """
        x = self._rect.x + self._mox
        y = self._rect.y + self._moy
        dxref, dyref = dx, dy

        if self.on_ground:  # at the start
            dy = 0
            while dx != 0 and world_collider.overlap(self.mask, (x + dx, y + dy)):  # try but can't move forward
                while dy > -abs(dx) - SCALE:  # try to climb stair (SCALE being the height of a stair)
                    dy -= 1
                    if not world_collider.overlap(self.mask, (x + dx, y + dy)):  #
                        break
                else:  # couldn't climb stair
                    dx -= sign(dx)
                    dy = 0
                    continue  # continue and break could be removed (it's only to save one call to an overlap already solved)
                break  # stop if slope was climbed
            # down stair magnet
            while dy < abs(dx) + SCALE and not self._is_on_ground_at(x + dx, y + dy, world_collider):
                dy += 1
        else:
            while dx != 0 and world_collider.overlap(self.mask, (x + dx, y + dy)):
                while dy != 0:
                    dy -= sign(dy)  #change how dy is tested to accept "air slopes" from -dy - SCALE to abs(dx) + SCALE?
                    if not world_collider.overlap(self.mask, (x + dx, y + dy)):
                        break
                else:
                    dx -= sign(dx)
                    dy = dyref
                    continue
                break
            else:  # dx is 0
                while dy != 0 and world_collider.overlap(self.mask, (x + dx, y + dy)):
                    dy -= sign(dy)

        if dx != dxref:
            self.collide_wall()
        if dy != dyref:  # landed or touched the ceiling:
            if dyref > 0:
                self.land()
            else:
                self.collide_vertical()
        self.move_x(dx)
        self.move_y(dy)
        return dx, dy


class PhysicsMixin(object):
    """
    Simple body with gravity and no rotation
    """

    def __init__(self, x: float = None, y: float = None, weight: float = None, g: float = None, max_y_speed: float = None, **kwargs):
        """
        :param x: position in pixel on map
        :param y: position in pixel on map
        :param weight:
        :param g: gravitational acceleration in pixel/second²

        """
        super(PhysicsMixin, self).__init__(**kwargs)
        self.pos_x = x
        self.pos_y = y
        self.weight = weight
        self.gravitational_acc = g if g else GRAVITY
        self.max_y_speed = max_y_speed

        self.vel_x = 0.
        self.vel_y = 0.

    def move_x(self, dx: float):
        self.pos_x += dx

    def move_y(self, dy: float):
        self.pos_y += dy

    def gravity(self, dt: float):
        self.vel_y += dt*self.gravitational_acc

    def physic_step(self, dt: float):
        self.gravity(dt)
        self.pos_x += dt*self.vel_x
        self.pos_y += dt*self.vel_y


class CharacterClass(MaskPlatformingMixin, PhysicsMixin, InputDispatcherMixin, DynamicObject):
    """Abstract class for character"""

    class Action(Enum):
        MOVE_LEFT = auto()
        MOVE_RIGHT = auto()
        JUMP = auto()

    def __init__(self, rect: pg.Rect, mask: pg.Mask, mask_offset: Tuple[int, int], played_map: MapCollider, team=None):
        super(CharacterClass, self).__init__(x=rect.x, y=rect.y, rect=rect, mask=mask, mask_offset=mask_offset)
        self.register_action(self.Action.MOVE_LEFT, lambda b: setattr(self, 'left', b))
        self.register_action(self.Action.MOVE_RIGHT, lambda b: setattr(self, 'right', b))
        self.register_action(self.Action.JUMP, self.set_jump)

        self.left = False
        self.right = False

        self.max_x_speed = None

        self.played_map = played_map
        self.team = team

    def move_x(self, dx: int):
        self.pos_x += dx
        self._rect.x += dx

    def move_y(self, dy: int):
        self.pos_y += dy
        self._rect.y += dy

    def collide_wall(self):
        self.vel_x = 0
        self.pos_x = self._rect.x

    def land(self):
        super(CharacterClass, self).land()
        if self.on_ground:
            self.collide_vertical()

    def collide_vertical(self):
        self.vel_y = 0
        self.pos_y = self._rect.y

    def update_movement(self, dt: float):
        # magic numbers fiesta!
        ground_multi = 2 if self.on_ground else 1.4
        threshold = 0.85*self.max_x_speed

        if self.left != self.right:
            x_sign = self.right - self.left
            if x_sign*self.vel_x < 0:
                self.vel_x *= pow(0.89, 60*dt)
            if x_sign*self.vel_x < 5*SCALE:
                self.vel_x += x_sign*dt*ground_multi*self.max_x_speed*1.5
            elif x_sign*self.vel_x < threshold:
                self.vel_x += x_sign*dt*ground_multi*threshold
                if self.on_ground and x_sign*self.vel_x >= threshold:
                    self.vel_x = x_sign*self.max_x_speed
            elif x_sign*self.vel_x < self.max_x_speed:
                if self.on_ground:
                    self.vel_x = x_sign*self.max_x_speed
        else:
            if self.on_ground:
                self.vel_x *= pow(0.89, 60*dt)
            else:
                if abs(self.vel_x) > threshold/2:
                    self.vel_x *= pow(0.96, 60*dt)
                else:
                    self.vel_x *= pow(1 - FRICTION, 60*dt)
            if abs(self.vel_x) < 0.008:
                self.vel_x = 0

        if self.vel_y > self.max_y_speed:
            self.vel_y = self.max_y_speed

    def update(self, dt: float, **kwargs):
        super(CharacterClass, self).update(dt, **kwargs)
        self.update_movement(dt)

        self.physic_step(dt)

        self.ground_check(self.played_map)
        self.pixel_platforming(*self._compute_deltas(dt), world_collider=self.played_map)


key_mapping = {
    pg.K_a: CharacterClass.Action.MOVE_LEFT,
    pg.K_d: CharacterClass.Action.MOVE_RIGHT,
    pg.K_w: CharacterClass.Action.JUMP}


def call_action(target: CharacterClass, key, value):
    try:
        action_name = key_mapping[key]
        target.dispatch_input({action_name: value})
    except KeyError:
        pass


class Clock(object):
    def __init__(self):
        self._current_time = time.perf_counter()

    def get_delta(self):
        current_time = time.perf_counter()
        delta = current_time - self._current_time
        if __debug__ and delta > 1/60:
            self._current_time += 1/60
            return 1/60
        self._current_time = current_time
        return delta


if __name__ == '__main__':
    pg.init()

    screen: pg.Surface = pg.display.set_mode((800, 600))
    level = GG2Map("../cp_mountainjazz.png")
    #level_image = pg.surfarray.make_surface(255 - level.mask.swapaxes(0, 1))
    level = MapCollider(level)
    level_image = level._mask.to_surface()
    player = CharacterClass(pg.Rect(1200, 600, 12, 50), pg.Mask((12, 50)), (0, 0), level)
    player.mask.fill()
    player._jump_speed = -360
    player.max_x_speed = 360
    player.max_y_speed = 800

    clock = Clock()
    running = True

    camera_rect = pg.Rect((0, 0), screen.get_size())
    while running:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                call_action(player, event.key, True)
            elif event.type == pg.KEYUP:
                call_action(player, event.key, False)
            elif event.type == pg.QUIT:
                running = False

        time_delta = clock.get_delta()
        player.update(time_delta)

        screen.fill((0, 0, 0))

        camera_rect.center = player._rect.center
        offset = camera_rect.topleft
        offset = [-1*_ for _ in offset]

        screen.blit(level_image, offset)
        r = player._rect.move(*offset)
        #r.x //= SCALE
        #r.y //= SCALE
        #r.width //= SCALE
        #r.height //= SCALE
        screen.fill((200, 30, 60), r)
        pg.display.set_caption(
            "p_rect:{}, p:{}, vx:{:.3f}, vy:{:.3f}, dt{:.3f}".format(
                player._rect.topleft, 'f', player.vel_x, player.vel_y, time_delta))
        pg.display.flip()
