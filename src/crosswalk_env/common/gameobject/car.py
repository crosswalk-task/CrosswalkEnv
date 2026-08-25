import numpy as np

from crosswalk_env.common.gameobject.game_object import GameObject
from crosswalk_env.common.utils import is_overlapping
from crosswalk_env.common.road import CrossWalk

class Car(GameObject):
    CAR_SPEEDS = [3.0, 3.5, 4.0, 4.5]  # NOTE: 2 is slightly faster than the player

    def __init__(self, uid, initial_x, start_row_idx, car_grid_height, road, map_grid_width, pix_square_size, steps_per_second, car_details, random):
        row_indices = []
        for i in range(car_grid_height):
            row_indices.append(start_row_idx + i)
        initial_y = np.mean(row_indices)
        cur_location = np.array([initial_x, initial_y], dtype=float)
        super().__init__(0, cur_location, pix_square_size, steps_per_second)
        self.uid = uid
        self.rows = row_indices
        self.map_grid_width = map_grid_width
        self.car_details = car_details
        self.random = random
        self.set_random_speed()
        self.road = road
        self.nearby_cars = [] # all the cars in the overlapping lane

    def get_cur_speed(self):
        return self.default_speed if self.is_moving else 0

    def get_hitbox(self):
        left_x, right_x = self.get_cur_x_pos()
        top_y, bottom_y = self.get_cur_y_pos()
        return [top_y, bottom_y, left_x, right_x]

    def get_cur_x_pos(self):
        left_x = self.cur_location[0] - (self.car_details.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_details.car_grid_width/2)
        return left_x, right_x

    def get_cur_y_pos(self):
        top_y = self.cur_location[1] - (self.car_details.car_grid_height/2)
        bottom_y = self.cur_location[1] + (self.car_details.car_grid_height/2)
        return top_y, bottom_y

    def set_random_speed(self):
        self.default_speed = self.random.choice(self.CAR_SPEEDS)
        if not self.car_details.go_right:
            self.default_speed *= -1

    def restart_if_needed(self):
        # teleport to start of the lane when reaching the end
        if self.car_details.go_right and self.cur_location[0] >= self.map_grid_width:
            self.cur_location[0] = 0
            self.set_random_speed()
        if not self.car_details.go_right and self.cur_location[0] <= 0:
            self.cur_location[0] = self.map_grid_width - 1
            self.set_random_speed()

    def update_target(self, dt):
        self.restart_if_needed()
        # stop in the current location when another car is in front of the car
        if len(self.nearby_cars) > 0:
            threshold = 0.1
            my_left, my_right = self.get_cur_x_pos()
            for other_car in self.nearby_cars:
                other_left, other_right = other_car.get_cur_x_pos()
                # stop when another car is in front of me
                if self.car_details.go_right:
                    front_dist = other_left - my_right
                else:
                    front_dist = my_left - other_right
                if 0 < front_dist <= threshold:
                    self.stop()
                    return

                # make the one in the back stop if the both cars are overlapping
                # make the one with lower uid stop if the front positions are the same
                if self.car_details.go_right:
                    front_diff = other_right - my_right
                else:
                    front_diff = my_left - other_left
                if front_diff > 0 or (front_diff == 0 and self.uid < other_car.uid):
                    if is_overlapping(my_left, my_right, other_left, other_right):
                        self.stop()
                        return

        # stop in front of the crosswalk when it is activated
        crosswalk = self.road.crosswalk
        if crosswalk is not None and crosswalk.is_active:
            car_left, car_right = self.get_cur_x_pos()
            cw_left, cw_right = crosswalk.get_activation_left_right()
            if self.car_details.go_right:
                front_dist = abs(cw_left - car_right)
            else:
                front_dist = abs(cw_right - car_left)
            if front_dist < CrossWalk.THRESHOLD_DISTANCE:
                if is_overlapping(car_left, car_right, cw_left, cw_right):
                    pass # keep going to clear up the crosswalk
                else:
                    self.stop()
                    return
        self.target_location[0] = self.cur_location[0] + (self.default_speed * dt / 1000)
        self.is_moving = True # moving unless stopped

    def render(self, background):
        left_x, _ = self.get_cur_x_pos()
        top_y, _ = self.get_cur_y_pos()
        left_x_pix = left_x * self.pix_square_size
        top_y_pix = top_y * self.pix_square_size
        adjustment = 0.5 * self.pix_square_size
        if self.car_details.image is not None:
            background.blit(self.car_details.image, (left_x_pix, top_y_pix + adjustment))

    def __str__(self):
        return f"Car(cur_pos=({self.cur_location[0], self.cur_location[1]}), crosswalk=({self.road.crosswalk}))"
