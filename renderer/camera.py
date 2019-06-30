import numpy as np
from glumpy import glm
from glumpy.app.window import key


class Camera(object):
    KEY_MAP = {  # TOTO: add an indirection level to mapping
        key.LEFT: "turning_left",
        key.RIGHT: "turning_right",
        key.UP: "moving_forward",
        key.DOWN: "moving_backward",
        ord('W'): "moving_up",
        ord('S'): "moving_down",
        ord('A'): "moving_left",
        ord('D'): "moving_right"
    }

    def __init__(self):
        self._position = np.zeros(3, dtype=np.float32)
        self._position[2] = -1.5
        self._yaw = 0  # around z
        self._pitch = 0  # inclination
        self._roll = 0

        self.turning_left = False
        self.turning_right = False

        self.moving_left = False
        self.moving_right = False
        self.moving_up = False
        self.moving_down = False
        self.moving_forward = False
        self.moving_backward = False

    def get_view(self):
        view = glm.scale(np.eye(4, dtype=np.float), 1, -1, 1)
        glm.translate(view, *(-self._position))  # shift the world so that the camera is at the origin
        glm.yrotate(view, self._yaw)  # rotate the world so that it faces the camera
        glm.xrotate(view, self._pitch)
        glm.zrotate(view, self._roll)
        glm.scale(view, 1, 1, -1)
        return view

    def set_state(self, state, new_value):
        try:
            attribute_name = self.KEY_MAP[state]
        except KeyError:
            return False
        setattr(self, attribute_name, new_value)
        return True

    def update(self, dt):

        # calculate the translation desired, apply rotation to it so that the translation
        # is applied from the camera points of view then apply it to the camera position
        translation = np.array([0, 0, 0, 1], dtype=np.float)
        if self.moving_right:
            translation[0] += dt
        if self.moving_left:
            translation[0] -= dt
        if self.moving_up:
            translation[1] += dt
        if self.moving_down:
            translation[1] -= dt
        if self.moving_forward:
            translation[2] += dt
        if self.moving_backward:
            translation[2] -= dt
        rotation = np.eye(4, dtype=np.float)
        glm.yrotate(rotation, self._yaw)
        glm.xrotate(rotation, self._pitch)
        glm.zrotate(rotation, self._roll)
        translation = np.matmul(rotation, translation)

        self._position += translation[:3]

        if self.turning_left:
            self._yaw -= 60*dt
        if self.turning_right:
            self._yaw += 60*dt

