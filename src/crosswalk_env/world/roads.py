from enum import Enum

import pygame

from crosswalk_env.common import get_max_car_grid_height, Road, CrossWalk

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class Roads:
    MIN_ROAD_SIZE = 2
    MAX_ROAD_SIZE = 4
    ROAD_GRAY_COLOR = (89, 89, 89)
    ROAD_WHITE_COLOR = (250, 250, 250)
    SAFE_WHITE_COLOR = (217, 217, 217)

    def __init__(self, agent, camera_width, map_grid_height, initial_height_padding, random):
        safe_end_lane_count = 1
        safe_start_lane_count = 1 + initial_height_padding # save safe zones near start lane
        max_car_grid_height = get_max_car_grid_height()
        if max_car_grid_height > self.MIN_ROAD_SIZE:
            raise Exception(f"maximum car height can not be bigger than minimum road size {max_car_grid_height} > {self.MIN_ROAD_SIZE}")

        row_types = [RowType.SAFE] * map_grid_height
        consecutive_danger_lanes = 0
        consecutive_safe_lanes = 1
        cur_row_idx = safe_end_lane_count
        while cur_row_idx < map_grid_height:
            # go back to add roads on too much consecutive safe zones
            if consecutive_safe_lanes >= self.MIN_ROAD_SIZE + 1:
                go_back = self.MIN_ROAD_SIZE
                cur_row_idx -= go_back
                consecutive_safe_lanes -= go_back

            available_rows = map_grid_height - safe_start_lane_count - cur_row_idx
            if available_rows <= 0: break 
            if available_rows >= self.MIN_ROAD_SIZE and consecutive_danger_lanes < self.MAX_ROAD_SIZE:
                if cur_row_idx == 1 or available_rows == self.MIN_ROAD_SIZE:
                    # reduce consecutive safe zones near the start and end
                    row_type = random.choice([RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                else:
                    row_type = random.choice([RowType.SAFE, RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                if row_type != RowType.SAFE:
                    for _ in range(self.MIN_ROAD_SIZE):
                        row_types[cur_row_idx] = row_type
                        consecutive_danger_lanes += 1
                        consecutive_safe_lanes = 0
                        cur_row_idx += 1
                    continue
            consecutive_danger_lanes = 0
            consecutive_safe_lanes += 1
            cur_row_idx += 1

        self.safe_row_idx_list = []
        for idx in range(len(row_types)):
            if row_types[idx] == RowType.SAFE:
                self.safe_row_idx_list.append(idx)

        self.lane_boundary_idx_list = []
        for idx in range(len(row_types)-1):
            if row_types[idx] != RowType.SAFE and row_types[idx+1] != RowType.SAFE:
                self.lane_boundary_idx_list.append(idx)

        self.other_direction_boundary_idx_list = []
        for idx in range(len(row_types)-1):
            if row_types[idx] == RowType.CAR_GOING_RIGHT and row_types[idx+1] == RowType.CAR_GOING_LEFT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)
            if row_types[idx] == RowType.CAR_GOING_LEFT and row_types[idx+1] == RowType.CAR_GOING_RIGHT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)

        uid = 0
        roads = []
        danger_start_idx = None
        for row_idx in range(len(row_types)):
            row_type = row_types[row_idx]
            if danger_start_idx is None:
                if row_type != RowType.SAFE:
                    danger_start_idx = row_idx
                continue
            if row_type == RowType.SAFE:
                uid += 1
                top_y, bottom_y = danger_start_idx, row_idx-1
                going_right_rows = [row_type == RowType.CAR_GOING_RIGHT for row_type in row_types[top_y:bottom_y+1]]
                roads.append(Road(uid, top_y, bottom_y, going_right_rows, random))
                danger_start_idx = None

        # add crosswalks on road
        crosswalk_uid = 0
        [agent_initial_col, _] = agent.cur_location
        buffer = int(camera_width * 0.3)
        left = random.random() >= 0.5
        for road in roads:
            if random.random() >= CrossWalk.RATIO: continue
            crosswalk_uid += 1
            crosswalk_range = (agent.MIN_X, max(agent.MIN_X+1, agent_initial_col - buffer)) if left else (min(agent.MAX_X-1, agent_initial_col + buffer), agent.MAX_X)
            left = not left
            col = random.integers(crosswalk_range[0], crosswalk_range[1])
            road.crosswalk = CrossWalk(crosswalk_uid, col, road.top_y, road.bottom_y)

        self.agent = agent
        self.elements = roads
        self.row_types = row_types
        self.activated_crosswalk_uid = 0 # NOTE: maximum of only 1 crosswalk can be activated

    def activate_crosswalks(self):
        self.activated_crosswalk_uid = 0
        agent_x, agent_y = self.agent.get_cur_location_rounded()
        for road in self.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            # deactivate right after completely crossing the road
            top_y = crosswalk.top_row - 1
            # activate before crossing the road
            bottom_y = crosswalk.end_row + 1 
            left_x, right_x = crosswalk.left_end, crosswalk.right_end
            crosswalk.is_active = (top_y < agent_y <= bottom_y) and (left_x <= agent_x <= right_x)
            if crosswalk.is_active:
                self.activated_crosswalk_uid = crosswalk.uid
                break

    def render(self, background, map_grid_width, pix_square_size):
        map_width = map_grid_width * pix_square_size
        background.fill(self.ROAD_GRAY_COLOR)
        for safe_row_idx in self.safe_row_idx_list:
            pygame.draw.rect(background, self.SAFE_WHITE_COLOR, (0, safe_row_idx * pix_square_size, map_width, pix_square_size + 1))

        for other_direction_idx in self.other_direction_boundary_idx_list:
            start_x = 0
            pygame.draw.line(
                background,
                self.ROAD_WHITE_COLOR,
                (start_x, pix_square_size * (other_direction_idx + 1)),
                (map_width, pix_square_size * (other_direction_idx + 1)),
                width=3,
            )

        for boundary_idx in self.lane_boundary_idx_list:
            for x in range(0, map_grid_width, 5):
                start_x = x * pix_square_size
                pygame.draw.line(
                    background,
                    self.ROAD_WHITE_COLOR,
                    (start_x, pix_square_size * (boundary_idx + 1)),
                    (start_x + 2 * pix_square_size, pix_square_size * (boundary_idx + 1)),
                    width=6,
                )

        # add crosswalks
        for road in self.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            start_x, end_x = crosswalk.get_activation_left_right()
            cw_width = (end_x - start_x) * pix_square_size
            start_x = start_x * pix_square_size
            top_y = road.top_y * pix_square_size
            bottom_y = (road.bottom_y + 1) * pix_square_size
            cw_height = bottom_y - top_y
            # NOTE: cover up background (-1 pixel at top and bottom, +pix_square_size at left and right)
            pygame.draw.rect(background, self.ROAD_GRAY_COLOR, (start_x - pix_square_size, top_y + 1, cw_width + pix_square_size * 2, cw_height - 1))

            stripe_count = 3 * (road.bottom_y - road.top_y + 1)
            stripe_thickness = pix_square_size / 3
            for i in range(stripe_count):
                for j in range(2):
                    if i % 2 == j: continue
                    stripe_rect = pygame.Rect(start_x + (cw_width/2 * j), top_y + (i * stripe_thickness), cw_width/2, stripe_thickness)
                    pygame.draw.rect(background, self.ROAD_WHITE_COLOR, stripe_rect)
