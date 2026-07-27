class GameObject:

    def __init__(self, default_speed, cur_location, pix_square_size, steps_per_second):
        self.default_speed = default_speed
        self.cur_location = cur_location
        self.target_location = self.cur_location
        self.pix_square_size = pix_square_size
        self.steps_per_second = steps_per_second
        self.is_moving = False

    def update_position(self, dt):
        tx, ty = self.target_location
        cx, cy = self.cur_location

        dx = tx - cx
        dy = ty - cy
        if abs(dx) < 0.1:
            self.cur_location[0] = tx
            dx = 0
        if abs(dy) < 0.1:
            self.cur_location[1] = ty
            dy = 0
        if dx == 0 and dy == 0: return

        if dx != 0 and dy != 0: # diagonal move
            dist = (dx ** 2 + dy ** 2) ** 0.5
        else: # straight move
            dist = max(abs(dx), abs(dy))
        step = abs(self.default_speed) * (dt / 1000.0)
        move = min(step, dist)
        ux = dx / dist
        uy = dy / dist

        self.cur_location[0] += ux * move
        self.cur_location[1] += uy * move
        self.is_moving = True

    def stop(self):
        self.target_location = self.cur_location
        self.is_moving = False

    def render(self, background):
        pass
