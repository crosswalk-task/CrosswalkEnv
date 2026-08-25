import os

import pygame
import numpy as np

from crosswalk_env.common import Action, ACTION_TO_DELTA, ACTION_DURATION
from crosswalk_env.common.gameobject.game_object import GameObject

AGENT_STATUS_ALIVE = "alive"
AGENT_STATUS_DEAD = "dead"

class Agent(GameObject):
    TARGET_LANE = 0

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, player_asset_info):
        fixed_speed = 2.5
        self.init_pos = (int(map_grid_width / 2), map_grid_height - 1)
        cur_location = np.array([self.init_pos[0], self.init_pos[1]], dtype=float)
        super().__init__(fixed_speed, cur_location, pix_square_size, steps_per_second)
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.is_dead = False
        self.MIN_X = agent_x_range[0]
        self.MAX_X = agent_x_range[1]
        self.last_direction = Action.DOWN.value
        self.images = player_asset_info["images"]
        self.sizes = player_asset_info["sizes"]
        self.mini_step_count = 0

    def get_hitbox(self):
        center_x = self.cur_location[0]
        center_y = self.cur_location[1]
        sub_dir = AGENT_STATUS_DEAD if self.is_dead else AGENT_STATUS_ALIVE
        direction = self.mini_step_count % ACTION_DURATION[self.last_direction]
        [player_width, player_height] = self.sizes[self.last_direction][sub_dir][direction]
        top_y = center_y - player_height/2
        bottom_y = top_y + player_height
        left_x = center_x - player_width/2
        right_x = left_x + player_width
        return [top_y, bottom_y, left_x, right_x]

    def get_cur_location_grid(self):
        return int(round(self.cur_location[0])), int(round(self.cur_location[1]))

    def get_cur_location_rounded(self):
        return round(self.cur_location[0], 2), round(self.cur_location[1], 2)

    def get_target_y(self):
        return round(self.target_location[1], 2)

    def update_target(self, action_value):
        if action_value != Action.NOTHING.value:
            self.last_direction = action_value
        delta = ACTION_TO_DELTA[action_value] * (self.default_speed / self.steps_per_second) * ACTION_DURATION[action_value]
        self.target_location = np.clip(self.cur_location + delta,
                                       [self.MIN_X, self.TARGET_LANE],
                                       [self.MAX_X, self.map_grid_height - 1])

    def set_dead(self):
        self.is_dead = True
        self.stop()

    def render(self, background):
        adjustment = 0.5 * self.pix_square_size
        agent_center_x = self.cur_location[0] * self.pix_square_size
        agent_center_y = self.cur_location[1] * self.pix_square_size + adjustment
        agent_position = (agent_center_x, agent_center_y)
        if self.images is not None:
            sub_dir = AGENT_STATUS_DEAD if self.is_dead else AGENT_STATUS_ALIVE
            direction = self.mini_step_count % ACTION_DURATION[self.last_direction]
            image = self.images[self.last_direction][sub_dir][direction]
            [agent_grid_width, agent_grid_height] = self.sizes[self.last_direction][sub_dir][direction]
            left_x_pix = agent_center_x - (agent_grid_width * self.pix_square_size / 2)
            top_y_pix = agent_center_y - (agent_grid_height * self.pix_square_size / 2)
            background.blit(image, (left_x_pix, top_y_pix))
        return agent_position

def load_player_asset_info(pix_square_size, render_sprite):
    directions = [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]
    size_ratio = {
        "player_UP_0_cropped.png": [57,73],
        "player_UP_1_cropped.png": [53,69],
        "player_UP_2_cropped.png": [57,73],
        "player_UP_3_cropped.png": [53,69],

        "player_DOWN_0_cropped.png": [57,73],
        "player_DOWN_1_cropped.png": [53,69],
        "player_DOWN_2_cropped.png": [57,73],
        "player_DOWN_3_cropped.png": [53,69],

        "player_RIGHT_0_cropped.png": [49,69],
        "player_RIGHT_1_cropped.png": [49,65],
        "player_RIGHT_2_cropped.png": [49,69],
        "player_RIGHT_3_cropped.png": [49,65],

        "player_LEFT_0_cropped.png": [49,69],
        "player_LEFT_1_cropped.png": [49,65],
        "player_LEFT_2_cropped.png": [49,69],
        "player_LEFT_3_cropped.png": [49,65],
    }
    images = {} if render_sprite else None
    sizes = {}
    for d in directions:
        if render_sprite:
            images[d.value] = {}
            images[d.value][AGENT_STATUS_ALIVE] = []
            images[d.value][AGENT_STATUS_DEAD] = []
        sizes[d.value] = {}
        sizes[d.value][AGENT_STATUS_ALIVE] = []
        sizes[d.value][AGENT_STATUS_DEAD] = []
        for i in range(4):
            for status in [AGENT_STATUS_ALIVE, AGENT_STATUS_DEAD]:
                file_name = f"player_{str(d)}_{i}_cropped.png"
                current_dir = os.path.dirname(os.path.abspath(__file__))
                image_path = os.path.join(current_dir, "../..", f"sprites/player_color/", file_name)
                [width_ratio, height_ratio] = size_ratio[file_name]
                agent_grid_height = (height_ratio / 140) * (0.4 if status == AGENT_STATUS_DEAD else 1)
                agent_grid_width = (width_ratio / 140) * (2.4 if status == AGENT_STATUS_DEAD else 1)
                if render_sprite:
                    object_image = pygame.image.load(image_path)
                    image = pygame.transform.scale(object_image, (agent_grid_width * pix_square_size, agent_grid_height * pix_square_size))
                    images[d.value][status].append(image)
                sizes[d.value][status].append([agent_grid_width, agent_grid_height])
    return {"images": images, "sizes": sizes}
