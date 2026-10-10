import gymnasium as gym

import pygame
import numpy as np

from crosswalk_env.world.world import World
from crosswalk_env.common import Car, load_player_asset_info, get_max_car_grid_width, load_car_details_dict, Action, ACTION_TO_DELTA, ACTION_DURATION, ACTION_LATENCY, is_overlapping, is_overlapping_rectangles
from crosswalk_env.common.gameobject.car_details import CarColorType

class CrosswalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    OFF_SCREEN_BLACK_COLOR = (0, 0, 0)
    UI_TEXT_WHITE_COLOR = (255, 255, 255)
    AGENT_DEAD_TEXT_COLOR = (255, 0, 0)
    SUCCESS_TEXT_COLOR = (0, 255, 0)
    TIME_OVER_TEXT_COLOR = (0, 0, 255)

    EXTRA_WIDTH = 500
    EXTRA_HEIGHT = 200

    def __init__(
            self,
            title="Crosswalk Task",
            width=25,
            height=25,
            initial_height_padding=0,
            camera_width=11,
            camera_height=7,
            steps_per_second=10,
            gamescreen_width_fixed=False,
            episode_duration_sec=30,
            time_left_warning_threshold_sec=10,
            basic_score_per_grid=50,
            bonus_score_per_sec=50,
            penalty_per_car_color={ "RED": 1000, "YELLOW": 500, "GREEN": 100 },
            gameover_screen_time=5000,
            fixed_episode_seed_range=None,
            render_mode=None,
            realtime=False
    ):
        if width < 12: raise Exception("minimum width is 13")
        if height < 6: raise Exception("minimum height is 6")
        if initial_height_padding < 0: raise Exception("initial_height_padding can not be negative")
        if camera_height < 7: raise Exception("minimum camera_height is 7")
        if camera_height%2 == 0: raise Exception("minimum camera_height should be an odd number")
        if time_left_warning_threshold_sec > 0 and episode_duration_sec < time_left_warning_threshold_sec: 
            raise Exception(f"episode duration ({episode_duration_sec} sec) should be longer than warning threshold ({time_left_warning_threshold_sec} sec)")

        valid_colors = {color.name for color in CarColorType}
        invalid_colors = set(penalty_per_car_color) - valid_colors
        if invalid_colors:
            raise ValueError(
                f"Invalid car colors: {invalid_colors}. "
                f"(only {', '.join(sorted(valid_colors))} are available)"
            )
        # NOTE: disable car colors without penalty configuration
        self.penalty_per_car_color = {
            color: penalty for color, penalty in penalty_per_car_color.items() if penalty > 0
        }

        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height
        self.initial_height_padding = initial_height_padding
        self.steps_per_second = steps_per_second
        self.step_ms =  1000 / self.steps_per_second # default: step once every 100ms
        self.realtime = realtime
        self.gameover_screen_time = gameover_screen_time
        self.metadata["render_fps"] = 60 # NOTE: must render multiple times between each step
        game_window_size = 2048
        self.pix_square_size = max(80, (game_window_size / max(self.map_grid_width, self.map_grid_height))) # The size of a single grid square in pixels
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_width_pixel = camera_width * self.pix_square_size
        self.camera_height_pixel = camera_height * self.pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size
        self.gamescreen_width_fixed = gamescreen_width_fixed
        assert fixed_episode_seed_range is None or len(fixed_episode_seed_range) > 0
        self.fixed_episode_seed_range = fixed_episode_seed_range
        self.fixed_episode_seed_idx = 0
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_sprite = render_mode is not None
        self.player_asset_info = load_player_asset_info(self.pix_square_size, self.render_sprite)
        self.car_details_dict = load_car_details_dict(self.penalty_per_car_color, self.pix_square_size, self.render_sprite)
        self.window = None
        self.clock = None

        # game info
        self.world = None
        self.cur_seed = None
        self.prev_rewards = 0 # sum of all the rewards from all the previous episodes
        self.cur_rewards = 0 # reward from the current ongoing episode
        self.best_rewards = 0
        self.GAME_TIME_MS = (episode_duration_sec + 1) * 1000
        self.time_left = self.GAME_TIME_MS
        self.time_left_warning_threshold_sec = time_left_warning_threshold_sec
        self.basic_score_per_grid = basic_score_per_grid
        self.bonus_score_per_sec = bonus_score_per_sec
        self.elapsed = 0
        self.game_over = False
        self.game_end_extra_score = 0
        self.real_time_step_passed = 0
        self.obs = None
        self.info = None
        
        self.max_car_penalty = max(penalty_per_car_color.values())
        self.max_car_speed = max(Car.CAR_SPEEDS)
        self.max_car_width = get_max_car_grid_width()
        agent_x_buffer = 2 + int(max(self.max_car_width, self.camera_width)/2)
        agent_min_x = (agent_x_buffer)
        agent_max_x = self.map_grid_width - 1 - agent_x_buffer
        self.agent_x_range = (agent_min_x, agent_max_x)

        self.action_space = gym.spaces.Discrete(5)
        self._define_observation_space()

    def reset(self, seed=None, options=None):
        # set seed for World initialization and car speed change
        self.cur_seed = seed
        if seed is None and self.fixed_episode_seed_range is not None:
            # use each seed in the fixed_episode_seed_range
            self.cur_seed = self.fixed_episode_seed_range[self.fixed_episode_seed_idx]
            self.fixed_episode_seed_idx = (self.fixed_episode_seed_idx + 1) % len(self.fixed_episode_seed_range)
        super().reset(seed=self.cur_seed)
        self.elapsed = 0

        self.best_rewards = max(self.best_rewards, self.cur_rewards)
        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0
        self.real_time_step_passed = 0

        self.world = World(
            self.agent_x_range,
            self.map_grid_width,
            self.map_grid_height,
            self.initial_height_padding,
            self.camera_width,
            self.pix_square_size,
            self.basic_score_per_grid,
            self.steps_per_second,
            self.penalty_per_car_color,
            self.np_random,
            self.player_asset_info,
            self.car_details_dict,
        )

        if self.render_sprite:
            self.render()

        self.obs = None
        self.info = None
        self._update_step_state()
        return self.obs, self.info

    def step(self, action):
        cumulative_reward = -self.world.calculate_cum_crossing_rewards()
        action_duration = ACTION_DURATION[action] + ACTION_LATENCY[action]
        self.real_time_step_passed += action_duration
        if not self.realtime:
            # interacting with the environment using RL algorithm
            self.world.agent.update_target(action)
            for _ in range(action_duration):
                reward, terminated = self._check_gameover()
                cumulative_reward += reward
                self.apply_time_and_render(self.step_ms)
                self.game_over = terminated
                if self.game_over: break
        else:
            # behaviour task for humans
            cur_count = 0
            self.world.agent.update_target(action)
            while True:
                if self.elapsed >= self.step_ms:
                    self.elapsed -= self.step_ms
                    cur_count += 1
                    if cur_count >= action_duration: break
                    reward, terminated = self._check_gameover()
                    cumulative_reward += reward
                    self.game_over = terminated
                    if self.game_over: break
                self.world.agent.mini_step_count = 0 if action == Action.NOTHING.value else cur_count
                dt = self.clock_tick()
                self.elapsed += dt
                self.apply_time_and_render(dt)

        # An episode is finished if the agent has reached the target lane
        if not self.world.agent.is_dead and self.world.target_lane_reached():
            self.game_over = True
            game_end_extra_score = self._get_time_left_sec()
            cumulative_reward += game_end_extra_score * self.bonus_score_per_sec
            self.game_end_extra_score = game_end_extra_score * self.bonus_score_per_sec

        cumulative_reward += self.world.calculate_cum_crossing_rewards()
        self.cur_rewards += cumulative_reward

        if self.game_over and self.realtime:
            self._render_game_over()
        
        self._update_step_state()
        return self.obs, float(cumulative_reward), self.game_over, False, self.info

    def _check_gameover(self):
        reward, terminated = 0, False
        agent_dead, death_penalty = self.world.check_and_update_agent_collision()
        if agent_dead:
            terminated = True
            reward -= death_penalty
            self.game_end_extra_score = -death_penalty
        elif self._get_time_left_sec() <= 0:
            self.world.agent.stop()
            terminated = True
        return reward, terminated

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def apply_time_and_render(self, dt):
        # NOTE: world should always keep moving continuously
        self.world.update_positions(dt)
        if not self.game_over:
            self.time_left -= dt
        self.render()

    def render(self):
        if self.render_mode is None: return
        total_window_width = self.camera_width_pixel + self.EXTRA_WIDTH
        total_window_height = self.camera_height_pixel + self.EXTRA_HEIGHT

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption(self.title)
                pygame.display.init()
                self.window = pygame.display.set_mode((total_window_width, total_window_height))
            elif self.render_mode == "rgb_array":
                self.window = pygame.Surface((total_window_width, total_window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas, rendered_agent_position = self.world.render()

        # clear background area
        bg_rect = pygame.Rect(0, 0, total_window_width, total_window_height)
        pygame.draw.rect(self.window, self.OFF_SCREEN_BLACK_COLOR, bg_rect)

        camera_rect = pygame.Rect(0, 0, self.camera_width_pixel, self.camera_height_pixel)
        if self.gamescreen_width_fixed:
            camera_rect.center = (self.world.agent.init_pos[0] * self.pix_square_size, rendered_agent_position[1] - (2 * self.pix_square_size))
        else:
            camera_rect.center = (rendered_agent_position[0], rendered_agent_position[1] - (2 * self.pix_square_size))
        # camera_rect.clamp_ip(canvas.get_rect()) # prevent camera from going out-of-bounds

        font_size_s = 32
        font_size_m = 48
        font_size_l = 80
        font_size_xl = 128
        center_x = total_window_width/2
        x_padding = 60
        left_ui_x = (self.EXTRA_WIDTH/2) - x_padding
        center_ui_x = center_x - x_padding
        right_ui_x = total_window_width - (self.EXTRA_WIDTH/2) - x_padding

        top_y = self.EXTRA_HEIGHT/2
        bottom_y = total_window_height - (self.EXTRA_HEIGHT/2) + 5
        left_game_x = self.EXTRA_WIDTH/2
        right_game_x = total_window_width - (self.EXTRA_WIDTH/2)
        top_game_y = self.EXTRA_HEIGHT/2
        dark_screen_alpha = 180 # 128 == 50% transparency

        # add game screen
        self.window.blit(canvas, (left_game_x, top_game_y), area=camera_rect)
        
        score_text_color = self.UI_TEXT_WHITE_COLOR
        if self.game_over:
            # add gameover screen
            dark_screen = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
            dark_screen.fill((0, 0, 0, dark_screen_alpha))
            self.window.blit(dark_screen, (left_game_x, top_game_y), area=camera_rect)

            if self.game_end_extra_score < 0:
                game_status_desc = "YOU DIED"
                game_status_color = self.AGENT_DEAD_TEXT_COLOR
                score_text_color = self.AGENT_DEAD_TEXT_COLOR
            elif self.game_end_extra_score > 0:
                game_status_desc = "BONUS"
                game_status_color = self.SUCCESS_TEXT_COLOR
                score_text_color = self.SUCCESS_TEXT_COLOR
            else:
                game_status_desc = "TIME OVER"
                game_status_color = self.TIME_OVER_TEXT_COLOR

            test_render_start_height = total_window_height/2
            self._render_text(center_x, test_render_start_height, game_status_desc, game_status_color, font_size_xl, order="center")
            test_render_start_height += font_size_xl/2

            if self.game_end_extra_score != 0:
                text = f"+{self.game_end_extra_score}" if self.game_end_extra_score > 0 else f"{self.game_end_extra_score}"
                self._render_text(center_x, test_render_start_height, text, game_status_color, font_size_l, order="center")
                test_render_start_height += font_size_l/2

            # show score results
            center_left_ui_x = left_game_x + 80
            center_right_ui_x = right_game_x - 130

            # update current score (include game_end_extra_score)
            self._render_text(center_left_ui_x, test_render_start_height, f"Current Score:", score_text_color, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{self.cur_rewards}", score_text_color, font_size_m, order="right")
            test_render_start_height += font_size_m

            # update best score
            best_renewed = self.cur_rewards > self.best_rewards
            best_score_text_color = self.UI_TEXT_WHITE_COLOR
            if best_renewed and self.game_end_extra_score > 0:
                best_score_text_color = self.SUCCESS_TEXT_COLOR
            best_score = self.cur_rewards if best_renewed else self.best_rewards
            self._render_text(center_left_ui_x, test_render_start_height, f"Best Score:", best_score_text_color, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{best_score}", best_score_text_color, font_size_m, order="right")
            if best_renewed:
                self._render_text(center_right_ui_x + 10, test_render_start_height, "(NEW!)", best_score_text_color, font_size_m)
            test_render_start_height += font_size_m

            # update total score
            self._render_text(center_left_ui_x, test_render_start_height, f"Total Score:", self.UI_TEXT_WHITE_COLOR, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{self.prev_rewards + self.cur_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_m, order="right")

            # update time left
            time_left_sec = self._get_time_left_sec()
            time_left_sec_color = self.UI_TEXT_WHITE_COLOR
            if self.time_left_warning_threshold_sec > 0 and time_left_sec <= self.time_left_warning_threshold_sec:
                time_left_sec_color = self.AGENT_DEAD_TEXT_COLOR
            self._render_text(right_game_x - 30, top_y + 30, f"Time Left: {time_left_sec}", time_left_sec_color, font_size_m, order="right")
        else:
            # unreachable area
            if not self.gamescreen_width_fixed:
                (agent_min_x, agent_max_x) = self.agent_x_range
                [agent_cur_x, _] = self.world.agent.cur_location
                game_screen_left_end = agent_cur_x - (self.camera_width // 2)
                game_screen_right_end = agent_cur_x + (self.camera_width // 2)
                buffer = 10
                if game_screen_left_end < agent_min_x:
                    unreachable_area_width = (agent_min_x - game_screen_left_end) * self.pix_square_size + buffer
                    dark_screen = pygame.Surface((unreachable_area_width, self.map_height), pygame.SRCALPHA)
                    dark_screen.fill((0, 0, 0, dark_screen_alpha))
                    self.window.blit(dark_screen, (left_game_x - buffer, top_game_y))
                elif agent_max_x < game_screen_right_end:
                    unreachable_area_width = (game_screen_right_end - agent_max_x) * self.pix_square_size + buffer
                    dark_screen = pygame.Surface((unreachable_area_width, self.map_height), pygame.SRCALPHA)
                    dark_screen.fill((0, 0, 0, dark_screen_alpha))
                    self.window.blit(dark_screen, (right_game_x - unreachable_area_width + buffer, top_game_y))

            # add ingame UI
            # update current score (include game_end_extra_score)
            self._render_text(center_ui_x, bottom_y, f"Current Score: {self.cur_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update total score: show the sum of previous scores (because it's confusing when both cur rewards and total rewards are constantly changing)
            self._render_text(right_ui_x, bottom_y, f"Total Score: {self.prev_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update best score
            self._render_text(left_ui_x, bottom_y, f"Best Score: {self.best_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update time left
            time_left_sec = self._get_time_left_sec()
            time_left_sec_color = self.UI_TEXT_WHITE_COLOR
            if self.time_left_warning_threshold_sec > 0 and time_left_sec <= self.time_left_warning_threshold_sec:
                time_left_sec_color = self.AGENT_DEAD_TEXT_COLOR

            self._render_text(right_ui_x, top_y - font_size_s - 5, f"Time Left: {time_left_sec}", time_left_sec_color, font_size_s)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _render_game_over(self):
        elapsed = 0
        total_elapsed = 0
        while total_elapsed < self.gameover_screen_time:
            dt = self.clock_tick()
            elapsed += dt
            total_elapsed += dt
            while elapsed >= self.step_ms:
                elapsed -= self.step_ms
                # NOTE: handle death during time over
                self.world.check_and_update_agent_collision()
            self.apply_time_and_render(dt)

    def _define_observation_space(self):
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0 for _ in range(113)]),
            high=np.array([1.0 for _ in range(113)]),
            dtype=np.float64
        )

    def _update_step_state(self):
        self._update_obs()
        self._update_info()

    def _update_obs(self, meta_info = True, tile_types = True, target_road_info = True, crosswalk_info = True, cars_info = True):
        obs = []
        agent_x, agent_y = self.world.agent.get_cur_location_grid()
        [_, _, agent_left_x, agent_right_x] = self.world.agent.get_hitbox()
        agent_width = agent_right_x - agent_left_x
        agent_y_range = [0, self.map_grid_height-1]
        y_top_visible_distance = self.camera_height//2 + 2
        y_bottom_visible_distance = self.camera_height//2 - 2
        grid_top_y = agent_y - y_top_visible_distance
        grid_bottom_y = grid_top_y + self.camera_height - 1
        x_visible_distance = self.camera_width//2
        if self.gamescreen_width_fixed:
            grid_x_left_end = self.world.agent.init_pos[0] - x_visible_distance
        else:
            grid_x_left_end = agent_x - x_visible_distance
        grid_x_right_end = grid_x_left_end + self.camera_width - 1

        visible_top_y_end = grid_top_y - 0.5
        visible_bottom_y_end = visible_top_y_end + self.camera_height
        visible_x_left_end = grid_x_left_end - 0.5
        visible_x_right_end = visible_x_left_end + self.camera_width
        game_screen_rect = [visible_top_y_end, visible_bottom_y_end, visible_x_left_end, visible_x_right_end]

        # Meta info
        if meta_info:
            time_left = 0
            if not self.game_over:
                time_left = max(0, (self.time_left - 1000) / (self.GAME_TIME_MS  - 1000))
            top_end_closeness = 0
            if grid_top_y < agent_y_range[0]:
                top_end_closeness = (y_top_visible_distance - (agent_y - agent_y_range[0])) / y_top_visible_distance
            bottom_end_closeness = 0
            if grid_bottom_y > agent_y_range[1]:
                bottom_end_closeness = (y_bottom_visible_distance - (agent_y_range[1] - agent_y)) / y_bottom_visible_distance
            left_end_closeness = 0
            if grid_x_left_end < self.agent_x_range[0]:
                left_end_closeness = (x_visible_distance - (agent_x - self.agent_x_range[0])) / x_visible_distance
            right_end_closeness = 0
            if self.agent_x_range[1] < grid_x_right_end:
                right_end_closeness = (x_visible_distance - (self.agent_x_range[1] - agent_x)) / x_visible_distance
            obs += [time_left, top_end_closeness, bottom_end_closeness, left_end_closeness, right_end_closeness]
        assert len(obs) == 5

        # Tile types near the agent
        if tile_types:
            is_dangerous = []
            is_crosswalk = []
            for action in [Action.NOTHING, Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]:
                [dx, dy] = ACTION_TO_DELTA[action.value]
                grid_x = agent_x + dx
                grid_y = agent_y + dy
                out_of_move_range = grid_y < agent_y_range[0] or grid_y > agent_y_range[1] or grid_x < self.agent_x_range[0] or grid_x > self.agent_x_range[1]
                if action != Action.RIGHT and action != Action.LEFT:
                    if out_of_move_range:
                        is_dangerous.append(0)
                    else:
                        is_dangerous.append(1 if self.world.danger_row[grid_y] else 0)
                if out_of_move_range:
                    is_crosswalk.append(0)
                else:
                    is_crosswalk.append(1 if self.world.crosswalk_map[grid_y][grid_x] else 0)
            obs += is_dangerous
            obs += is_crosswalk
        assert len(obs) == 13

        # Target road info
        self.prev_road = None # the road that the agent previously crossed
        self.target_road = None # the road that the agent is crossing or should cross
        self.target_road_crosswalk_visible = False
        for i in range(len(self.world.roads.elements), 0, -1):
            road = self.world.roads.elements[i-1]
            if road.top_y > agent_y:
                self.prev_road = road
                continue # already crossed
            self.target_road = road
            break
        if target_road_info:
            penalty = 0.0
            dist_end_of_target_road = 1.0
            dist_start_of_target_road = 0.0
            visible_car_count = 0
            dangerous_car_counts = [0, 0, 0, 0, 0, 0]
            if self.target_road is not None:
                road = self.target_road
                if road.crosswalk is not None:
                    if is_overlapping(road.crosswalk.left_end, road.crosswalk.right_end, grid_x_left_end, grid_x_right_end):
                        self.target_road_crosswalk_visible = True
                penalty = road.penalty / self.max_car_penalty
                is_crossing = road.bottom_y >= agent_y >= road.top_y
                if is_crossing:
                    dist_end_of_target_road = (agent_y - road.top_y) / 4
                    dist_start_of_target_road = (road.bottom_y - agent_y + 1) / 4
                cars = self.world.cars.road_uid_to_cars_dict[road.uid]
                for car in cars:
                    car_rect = car.get_hitbox()
                    if not is_overlapping_rectangles(car_rect, game_screen_rect): continue # filter out of sight cars
                    visible_car_count += 1
                    [_, _, car_left_x, car_right_x] = car_rect
                    is_left = car_left_x <= agent_x
                    is_right = agent_x <= car_right_x
                    going_right = car.car_details.go_right
                    if (is_left and not going_right) or (is_right and going_right): continue # filter safe cars
                    car_top_row, car_bottom_row = min(car.rows), max(car.rows)
                    car_in_front = car_top_row < agent_y
                    car_same_row = car_top_row == agent_y or agent_y == car_bottom_row
                    car_in_back = agent_y < car_bottom_row
                    if car_in_front and is_left:
                        dangerous_car_counts[0] += 1 # NW
                    if car_in_front and is_right:
                        dangerous_car_counts[1] += 1 # NE
                    if car_in_back and is_left:
                        dangerous_car_counts[2] += 1 # SE
                    if car_in_back and is_right:
                        dangerous_car_counts[3] += 1 # SW
                    if car_same_row and is_left:
                        dangerous_car_counts[4] += 1 # left
                    if car_same_row and is_right:
                        dangerous_car_counts[5] += 1 # right
                # NOTE: each road can have maximum of 6 cars
                visible_car_count /= 6
                dangerous_car_counts = [cnt / 6 for cnt in dangerous_car_counts]
            crosswalk_visible = 1.0 if self.target_road_crosswalk_visible else 0.0
            obs += [crosswalk_visible, penalty, dist_end_of_target_road, dist_start_of_target_road, visible_car_count]
            obs += dangerous_car_counts
        assert len(obs) == 24

        # Distance from crosswalks
        if crosswalk_info:
            crosswalk_closeness_list = []
            for i in range(len(self.world.roads.elements), 0, -1):
                road = self.world.roads.elements[i-1]
                if road.top_y > agent_y: continue # already crossed
                if road.bottom_y < grid_top_y: break # out of sight
                if road.crosswalk is None: continue

                crosswalk_closeness = [0, 0, 0] # up, right, left
                if road.crosswalk.top_row <= agent_y <= road.crosswalk.end_row:
                    crosswalk_closeness[0] = 1
                elif agent_y >= road.crosswalk.end_row:
                    if road.crosswalk.end_row >= grid_top_y:
                        crosswalk_closeness[0] = (y_top_visible_distance - (agent_y - road.crosswalk.end_row)) / (y_top_visible_distance)

                if road.crosswalk.left_end <= agent_x <= road.crosswalk.right_end:
                    crosswalk_closeness[1] = 1
                    crosswalk_closeness[2] = 1
                elif agent_x < road.crosswalk.left_end:
                    if road.crosswalk.left_end <= grid_x_right_end:
                        crosswalk_closeness[1] = (x_visible_distance + 1 - (road.crosswalk.left_end - agent_x)) / (x_visible_distance + 1)
                elif road.crosswalk.right_end < agent_x:
                    if grid_x_left_end <= road.crosswalk.right_end:
                        crosswalk_closeness[2] = (x_visible_distance + 1 - (agent_x - road.crosswalk.right_end)) / (x_visible_distance + 1)

                crosswalk_closeness_list += crosswalk_closeness
                if len(crosswalk_closeness_list) == 3 * 3: break

            while len(crosswalk_closeness_list) < 3 * 3:
                crosswalk_closeness_list += [0.0, 0.0, 0.0] # fill up defaults
            obs += crosswalk_closeness_list
        assert len(obs) == 33

        # Cars
        if cars_info:
            car_in_adjecent_tiles = []
            for i in range(8):
                # NW, N, NE, E, SE, S, SW, W
                dx = [-1, 0, 1, 1, 1, 0, -1, -1] 
                dy = [-1, -1, -1, 0, 1, 1, 1, 0]
                grid_x = agent_x + dx[i]
                grid_y = agent_y + dy[i]
                tile_rect = [grid_y - 0.5, grid_y + 0.5, grid_x - 0.5, grid_x + 0.5]
                car_in_tile = False
                if grid_y in self.world.row_to_cars_dict:
                    for car in self.world.row_to_cars_dict[grid_y]:
                        if is_overlapping_rectangles(tile_rect, car.get_hitbox()):
                            car_in_tile = True
                            break
                car_in_adjecent_tiles.append(1.0 if car_in_tile else 0.0)
            obs += car_in_adjecent_tiles

            dangerous_car_info_list = []
            visible_left_width = agent_left_x - visible_x_left_end
            visible_right_width = visible_x_right_end - agent_right_x
            # only consider until 4 rows ahead of agent since maximum road size is 4
            for grid_y in range(agent_y - 4, grid_bottom_y + 1):
                left_cars = []
                right_cars = []
                # only consider the roads near the agent
                is_prev_road = self.prev_road is not None and self.prev_road.top_y <= grid_y <= self.prev_road.bottom_y
                is_target_road = self.target_road is not None and self.target_road.top_y <= grid_y <= self.target_road.bottom_y
                if is_prev_road or is_target_road:
                    for car in self.world.row_to_cars_dict[grid_y]:
                        car_left_x, car_right_x = car.get_cur_x_pos()
                        if car_right_x < visible_x_left_end: continue # filter out of sight
                        if visible_x_right_end < car_left_x: continue # filter out of sight
                        car_left_x = max(car_left_x, visible_x_left_end)
                        car_right_x = min(car_right_x, visible_x_right_end)
                        car_speed = abs(car.default_speed) / self.max_car_speed if car.is_moving else 0
                        # cars on the left side and coming toward the agent
                        if car_left_x <= agent_right_x and car.car_details.go_right:
                            if car_right_x < agent_left_x:
                                # [car] < [agent]
                                car_head_closeness = (visible_left_width - (agent_left_x - car_right_x)) / visible_left_width
                                car_tail_distance = 0.0
                            else:
                                # car overlapping with the agent
                                car_head_closeness = 1.0
                                car_tail_distance = (agent_right_x - car_left_x) / (self.max_car_width + agent_width)
                            if car_left_x < agent_left_x:
                                # [car]=>
                                #    [agent]
                                car_tail_closeness = (visible_left_width - (agent_left_x - car_left_x)) / visible_left_width
                            else:
                                #   =>[car]
                                # [agent]
                                car_tail_closeness = 1.0
                            left_cars.append([car_head_closeness, car_tail_closeness, car_tail_distance, car_speed])
                        # cars on the right side and coming toward the agent
                        if agent_left_x <= car_right_x and not car.car_details.go_right:
                            if agent_right_x < car_left_x:
                                # [agent] < [car]
                                car_head_closeness = (visible_right_width - (car_left_x - agent_right_x)) / visible_right_width
                                car_tail_distance = 0.0
                            else:
                                # car overlapping with the agent
                                car_head_closeness = 1.0
                                car_tail_distance = (car_right_x - agent_left_x) / (self.max_car_width + agent_width)
                            if agent_right_x < car_right_x:
                                #   <=[car]
                                # [agent]
                                car_tail_closeness = (visible_right_width - (car_right_x - agent_right_x)) / visible_right_width
                            else:
                                # [car]<=
                                #   [agent]
                                car_tail_closeness = 1.0
                            right_cars.append([car_head_closeness, car_tail_closeness, car_tail_distance, car_speed])
                left_cars.sort(key=lambda x: x[0], reverse=True)
                right_cars.sort(key=lambda x: x[0], reverse=True)
                while len(left_cars) < 2:
                    left_cars.append([0.0, 0.0, 0.0, 0.0])
                while len(right_cars) < 2:
                    right_cars.append([0.0, 0.0, 0.0, 0.0])
                # consider the position and speed of 2 cars on each side 
                dangerous_car_info_list += left_cars[0] # [head closeness, tail closeness, tail distance, speed]
                dangerous_car_info_list += right_cars[0] # [head closeness, tail closeness, tail distance, speed]
                dangerous_car_info_list += [left_cars[1][0], left_cars[1][1]] # [head closeness, tail closeness]
                dangerous_car_info_list += [right_cars[1][0], right_cars[1][1]] # [head closeness, tail closeness]
            obs += dangerous_car_info_list
        assert len(obs) == 113
        assert max(obs) <= 1.0 and min(obs) >= 0.0
        self.obs = np.array(obs)

    def _update_info(self):
        agent = self.world.agent
        [agent_top_y, agent_bottom_y, agent_left_x, agent_right_x] = agent.get_hitbox()
        visible_distances, game_screen_grid_rect, game_screen_rect = self._game_screen_relative_positions()
        if self.info is None:
            agent_y_range = [agent.TARGET_LANE, self.map_grid_height - 1]
            [grid_y_top_visible_distance, grid_y_bottom_visible_distance, grid_x_visible_distance] = visible_distances
            agent_metadata = {
                "initial_pos": agent.init_pos,
                "x_range": self.agent_x_range,
                "y_range": agent_y_range,
                "speed": agent.default_speed,
                "grid_y_top_visible_distance": grid_y_top_visible_distance,
                "grid_y_bottom_visible_distance": grid_y_bottom_visible_distance,
                "grid_x_visible_distance": grid_x_visible_distance,
            }
            roads_metadata = {
                "is_dangerous_row": self.world.danger_row,
                "roads": [],
            }
            for i in range(len(self.world.roads.elements), 0, -1):
                road = self.world.roads.elements[i-1]
                roads_metadata["roads"].append({
                    "uid": road.uid,
                    "top_y": road.top_y,
                    "bottom_y": road.bottom_y,
                    "going_right": road.going_right,
                    "car_color_type": road.car_color_type.value, # default: 0=GREEN, 1=YELLOW, 2=RED
                    "penalty": road.penalty, # default: 100=GREEN, 500=YELLOW, 1000=RED
                    "crosswalk": None if road.crosswalk is None else {"uid": road.crosswalk.uid, "col": road.crosswalk.col},
                })
            cars_metadata = []
            for car in self.world.cars.elements:
                cars_metadata.append({
                    "uid": car.uid,
                    "road_uid": car.road.uid,
                    "rows": car.rows,
                    "car_type": car.car_details.car_type,
                    "penalty": car.car_details.penalty,
                    "go_right" :car.car_details.go_right,
                    "car_grid_height": car.car_details.car_grid_height,
                    "car_grid_width": car.car_details.car_grid_width,
                })
            self.info = {
                "env_configuration": {
                    "map_grid_width": self.map_grid_width,
                    "map_grid_height": self.map_grid_height,
                    "steps_per_second": self.steps_per_second,
                    "step_ms": self.step_ms,
                    "realtime": self.realtime,
                    "gameover_screen_time": self.gameover_screen_time,
                    "render_fps": self.metadata["render_fps"],
                    "pix_square_size": self.pix_square_size,
                    "camera_width": self.camera_width,
                    "camera_height": self.camera_height,
                    "gamescreen_width_fixed": self.gamescreen_width_fixed,
                    "render_mode": self.render_mode,
                },
                "episode_configuration": {
                    "seed": self.cur_seed,
                    "agent": agent_metadata,
                    "roads": roads_metadata,
                    "cars": cars_metadata,
                },
                "play_infos": None,
            }

        agent_grid_x, agent_grid_y = agent.get_cur_location_grid()
        [grid_top_y, grid_bottom_y, grid_x_left_end, grid_x_right_end] = game_screen_grid_rect
        [visible_top_y_end, visible_bottom_y_end, visible_x_left_end, visible_x_right_end] = game_screen_rect

        target_road_info = {}
        if self.target_road is not None:
            target_road_info = {
                "target_road_uid": self.target_road.uid,
                "road_size": self.target_road.bottom_y - self.target_road.top_y + 1, 
                "penalty": self.target_road.penalty,
                "has_crosswalk": self.target_road.crosswalk is not None,
            }

        visible_cars_per_lane = {}
        # only consider until 4 rows ahead of agent since maximum road size is 4
        for grid_y in range(agent_grid_y - 4, grid_bottom_y + 1):
            # only consider the roads near the agent
            is_prev_road = self.prev_road is not None and self.prev_road.top_y <= grid_y <= self.prev_road.bottom_y
            is_target_road = self.target_road is not None and self.target_road.top_y <= grid_y <= self.target_road.bottom_y
            if not is_prev_road and not is_target_road: continue
            cars = []
            for car in self.world.row_to_cars_dict[grid_y]:
                [car_top_y, car_bottom_y, car_left_x, car_right_x] = car.get_hitbox()
                if car_right_x < visible_x_left_end: continue # filter out of sight
                if visible_x_right_end < car_left_x: continue # filter out of sight
                car_left_x = max(car_left_x, visible_x_left_end)
                car_right_x = min(car_right_x, visible_x_right_end)
                car_speed = abs(car.default_speed) / self.max_car_speed if car.is_moving else 0
                car_going_right = car.car_details.go_right
                cars.append({
                    "top_y": car_top_y,
                    "bottom_y": car_bottom_y,
                    "left_x": car_left_x,
                    "right_x": car_right_x,
                    "speed": car_speed,
                    "going_right": car_going_right,
                })
            visible_cars_per_lane[grid_y] = cars

        self.info["play_infos"] = {
            "game_info": {
                "time_left": self.time_left,
                "real_time_step_passed": self.real_time_step_passed,
                "game_end_extra_score":  self.game_end_extra_score,
                "cur_episode_score": self.cur_rewards,
                "total_score": self._total_rewards(),
            },
            "map": {
                # game screen grid rectangle
                "grid_top_y": grid_top_y,
                "grid_bottom_y": grid_bottom_y,
                "grid_x_left_end": grid_x_left_end,
                "grid_x_right_end": grid_x_right_end,

                # game screen rectangle
                "visible_top_y_end": visible_top_y_end, 
                "visible_bottom_y_end": visible_bottom_y_end, 
                "visible_x_left_end": visible_x_left_end, 
                "visible_x_right_end": visible_x_right_end,

                "target_road_uid": self.target_road.uid if self.target_road is not None else None,
                "target_road_crosswalk_visible": self.target_road_crosswalk_visible,
                "activated_crosswalk_uid": self.world.roads.activated_crosswalk_uid,
            },
            "agent": {
                "grid_x": agent_grid_x,
                "grid_y": agent_grid_y,
                "cur_location": agent.cur_location,
                "target_location": agent.target_location,
                "facing_direction": agent.last_direction,
                # NOTE: hitbox = [top_y, bottom_y, left_x, right_x]; agent_width = agent_right_x - agent_left_x
                "top_y": agent_top_y,
                "bottom_y": agent_bottom_y,
                "left_x": agent_left_x,
                "right_x": agent_right_x, 
                "dead": agent.is_dead,
            },
            "target_road": target_road_info,
            "visible_cars": visible_cars_per_lane,
        }

    def _game_screen_relative_positions(self):
        agent = self.world.agent
        agent_grid_x, agent_grid_y = agent.get_cur_location_grid()
        grid_y_top_visible_distance = self.camera_height//2 + 2
        grid_y_bottom_visible_distance = self.camera_height//2 - 2
        grid_x_visible_distance = self.camera_width//2
        visible_distances = [grid_y_top_visible_distance, grid_y_bottom_visible_distance, grid_x_visible_distance]

        # game screen grid rectangle
        grid_top_y = agent_grid_y - grid_y_top_visible_distance
        grid_bottom_y = grid_top_y + self.camera_height - 1
        if self.gamescreen_width_fixed:
            grid_x_left_end = agent.init_pos[0] - grid_x_visible_distance
        else:
            grid_x_left_end = agent_grid_x - grid_x_visible_distance
        grid_x_right_end = grid_x_left_end + self.camera_width - 1
        game_screen_grid_rect = [grid_top_y, grid_bottom_y, grid_x_left_end, grid_x_right_end]

        # game screen rectangle
        visible_top_y_end = grid_top_y - 0.5
        visible_bottom_y_end = visible_top_y_end + self.camera_height
        visible_x_left_end = grid_x_left_end - 0.5
        visible_x_right_end = visible_x_left_end + self.camera_width

        game_screen_rect = [visible_top_y_end, visible_bottom_y_end, visible_x_left_end, visible_x_right_end]
        return visible_distances, game_screen_grid_rect, game_screen_rect

    def _total_rewards(self):
        return self.prev_rewards + self.cur_rewards

    def _get_time_left_sec(self):
        return int(self.time_left/1000)

    def _render_text(self, x, y, text, color, font_size, order=None):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        if order == "center":
            text_location = text_surface.get_rect(center=(x, y))
        elif order == "left":
            text_location = text_surface.get_rect(topleft=(x, y))   
        elif order == "right":
            text_location = text_surface.get_rect(topright=(x, y))            
        else:
            text_location = (x, y)
        self.window.blit(text_surface, text_location)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
