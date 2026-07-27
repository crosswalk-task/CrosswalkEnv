import pygame

from crosswalk_env.common import Agent, is_overlapping_rectangles
from crosswalk_env.world.cars import Cars
from crosswalk_env.world.roads import Roads

class World:
    UP_REWARD_PER_UNIT = 5

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, initial_height_padding, camera_width, pix_square_size, steps_per_second, random, debug, player_asset_info, car_details_dict):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size

        self.steps_per_second = steps_per_second

        self.agent = Agent(agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, player_asset_info, debug)
        self.initial_agent_y = self.agent.cur_location[1]
        self.roads = Roads(self.agent, camera_width, map_grid_height, initial_height_padding, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, car_details_dict, random)

        self.row_to_cars_dict = {}
        for car in self.cars.elements:
            for row in car.rows:
                if row not in self.row_to_cars_dict:
                    self.row_to_cars_dict[row] = []
                self.row_to_cars_dict[row].append(car)

        self.danger_row = [True for _ in range(map_grid_height)]
        for y in self.roads.safe_row_idx_list:
            self.danger_row[y] = False # safe

        self.crosswalk_map = [[False for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for road in self.roads.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            for y in range(crosswalk.top_row, crosswalk.end_row+2):
                for x in range(crosswalk.left_end, crosswalk.right_end+1):
                    self.crosswalk_map[y][x] = True # crosswalk

        reward_y = [False for _ in range(map_grid_height)]
        reward_y[Agent.TARGET_LANE] = True
        self.reward_per_y = [0 for _ in range(map_grid_height)]
        for road in self.roads.elements:
            road_crossed_y = road.top_y - 1
            reward_y[road_crossed_y] = True
            cum_reward = int(self.steps_per_second * (self.initial_agent_y - initial_height_padding - road_crossed_y)) * self.UP_REWARD_PER_UNIT
            self.reward_per_y[road_crossed_y] = cum_reward
        for y in range(map_grid_height-1, 0, -1):
            self.reward_per_y[y-1] = max(self.reward_per_y[y], self.reward_per_y[y-1])

    def check_and_update_agent_collision(self):
        agent_rect = self.agent.get_hitbox()
        for car in self.cars.elements:
            car_rectangle = car.get_hitbox()
            if is_overlapping_rectangles(agent_rect, car_rectangle):
                self.agent.set_dead()
                return True, car.car_details.penalty.value
        return False, 0

    def target_lane_reached(self):
        _, agent_y = self.agent.get_cur_location_grid()
        return agent_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.roads.activate_crosswalks()
        self.cars.update_positions(dt)

    def calculate_cum_crossing_rewards(self):
        _, agent_y = self.agent.get_cur_location_grid()
        return self.reward_per_y[agent_y]

    def render(self):
        map_width = self.map_grid_width * self.pix_square_size
        map_height = self.map_grid_height * self.pix_square_size
        canvas = pygame.Surface((map_width, map_height))
        self.roads.render(canvas, self.map_grid_width, self.pix_square_size)
        rendered_agent_position = self.agent.render(canvas)
        self.cars.render(canvas)
        return canvas, rendered_agent_position
