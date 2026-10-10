import os
import time
from datetime import datetime

import csv

import pygame
import argparse
import numpy as np

from crosswalk_env.environment import CrosswalkEnv
from crosswalk_env.common.action import Action

KEY_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_LEFT: Action.LEFT,

    pygame.K_w: Action.UP,
    pygame.K_s: Action.DOWN,
    pygame.K_d: Action.RIGHT,
    pygame.K_a: Action.LEFT,
}

def _play_episode(env, episode_seed):
    obs, info = env.reset(seed=episode_seed)
    episode_metadata = {"env_configuration": info["env_configuration"], "episode_configuration": info["episode_configuration"]}
    play_infos = [info["play_infos"]]
    observations = [obs]
    actions = []
    rewards = []

    pygame.event.get() # clear previous key presses
    last_action = Action.NOTHING
    while True:
        obs, reward, terminated, truncated, info = env.step(last_action.value)
        play_infos.append(info["play_infos"])
        observations.append(obs)
        actions.append(last_action.value)
        rewards.append(reward)
        if terminated or truncated: return False, episode_metadata, observations, actions, rewards, play_infos
        last_action = Action.NOTHING

        # check the keys that are being pressed
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True, episode_metadata, observations, actions, rewards, play_infos

            if last_action == Action.NOTHING:
                # instant start if the player started pressing a key
                if event.type == pygame.KEYDOWN:
                    last_action = KEY_ACTION.get(event.key, Action.NOTHING)
                    if last_action != Action.NOTHING: break
            else:
                # instant stop if the player stopped pressing a key
                if event.type == pygame.KEYUP and KEY_ACTION.get(event.key) == last_action:
                    last_action = Action.NOTHING
                    break

def _play_session(
        save_dir,
        session_id,
        max_seconds,
        max_episodes=-1,
        episode_duration_sec=30,
        time_left_warning_threshold_sec=10,
        basic_score_per_grid=50,
        bonus_score_per_sec=50,
        red_car_penalty=1000,
        yellow_car_penalty=500,
        green_car_penalty=100,
        base_seed=0,
    ):
    env = CrosswalkEnv(
        episode_duration_sec=episode_duration_sec,
        time_left_warning_threshold_sec=time_left_warning_threshold_sec,
        basic_score_per_grid=basic_score_per_grid,
        bonus_score_per_sec=bonus_score_per_sec,
        penalty_per_car_color={ 
            "RED": red_car_penalty,
            "YELLOW": yellow_car_penalty,
            "GREEN": green_car_penalty,
        },
        render_mode="human",
        realtime=True
    )
    start_timestamp = int(time.time())
    episode_count = 0
    while True:
        if max_episodes > 0 and episode_count >= max_episodes: break
        episode_count += 1
        if max_seconds > 0:
            time_passed = int(time.time()) - start_timestamp
            if time_passed >= max_seconds: break
        # NOTE: assume that each session lasts less than 1000 episodes
        episode_id = session_id * 1000 + episode_count 
        episode_seed = base_seed + episode_id
        quit_game, episode_metadata, observations, actions, rewards, play_infos = _play_episode(env, episode_seed)
        if quit_game: break
        if save_dir is not None:
            os.makedirs(f"{save_dir}/{episode_id:04d}", exist_ok=True)
            np.save(f"{save_dir}/{episode_id:04d}/observations.npy", np.array(observations))
            np.save(f"{save_dir}/{episode_id:04d}/actions.npy", np.array(actions))
            np.save(f"{save_dir}/{episode_id:04d}/rewards.npy", np.array(rewards))
            np.save(f"{save_dir}/{episode_id:04d}/episode_metadata.npy", np.array([episode_metadata], dtype=object), allow_pickle=True)
            np.save(f"{save_dir}/{episode_id:04d}/play_infos.npy", np.array(play_infos, dtype=object), allow_pickle=True)
    env.close()

def run_session(
        subj_id=1,
        session_id=1,
        max_episodes=-1,
        max_seconds=1200,
        episode_duration_sec=30,
        time_left_warning_threshold_sec=10,
        basic_score_per_grid=50,
        bonus_score_per_sec=50,
        red_car_penalty=1000,
        yellow_car_penalty=500,
        green_car_penalty=100,
        seed=0,
        save=True,
        base_save_dir="data",
    ):
    if not save:
        _play_session(
            save_dir=None,
            session_id=session_id,
            max_seconds=max_seconds,
            max_episodes=max_episodes,
            episode_duration_sec=episode_duration_sec,
            time_left_warning_threshold_sec=time_left_warning_threshold_sec,
            basic_score_per_grid=basic_score_per_grid,
            bonus_score_per_sec=bonus_score_per_sec,
            red_car_penalty=red_car_penalty,
            yellow_car_penalty=yellow_car_penalty,
            green_car_penalty=green_car_penalty,
            base_seed=seed,
        )
        return

    save_dir = os.path.join(base_save_dir, f"{subj_id}")
    os.makedirs(save_dir, exist_ok=True)

    session_log_path = os.path.join(base_save_dir, "session_log.csv")
    file_exists = os.path.exists(session_log_path)
    with open(session_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["subject_id", "session_id", "seed", "max_episodes", "max_seconds", "play_start_time"])
        start_time = datetime.now()
        writer.writerow([subj_id, session_id, seed, max_episodes, max_seconds, start_time.isoformat(timespec="seconds")])

    _play_session(
        save_dir=save_dir,
        session_id=session_id,
        max_episodes=max_episodes,
        max_seconds=max_seconds,
        episode_duration_sec=episode_duration_sec,
        time_left_warning_threshold_sec=time_left_warning_threshold_sec,
        basic_score_per_grid=basic_score_per_grid,
        bonus_score_per_sec=bonus_score_per_sec,
        red_car_penalty=red_car_penalty,
        yellow_car_penalty=yellow_car_penalty,
        green_car_penalty=green_car_penalty,
        base_seed=seed,
    )

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument(
        '--subj-id',
        type=int,
        default=1,
        help='subject ID',
    )
    arg_parser.add_argument(
        '--session-id',
        type=int,
        default=0,
        help='session ID',
    )
    arg_parser.add_argument(
        '--max-episodes',
        type=int,
        default=-1,
        help='total number of episodes',
    )
    arg_parser.add_argument(
        '--max-seconds',
        type=int,
        default=1200,
        help='ends after reaching max seconds (default: 20 min)',
    )
    arg_parser.add_argument(
        '--episode-duration-sec', 
        type=int, 
        default=30, 
        help='maximum time limit for each episode (default: 30 seconds)',
    )
    arg_parser.add_argument(
        '--time-left-warning-threshold-sec',
        type=int,
        default=10,
        help='highlight the remaining time in red when it reaches this threshold (default: 10 seconds; set to 0 to disable)',
    )

    arg_parser.add_argument(
        '--basic-score-per-grid',
        type=int,
        default=50,
        help='basic score per unit of upward movement (default: 50)',
    )
    arg_parser.add_argument(
        '--bonus-score-per-sec',
        type=int,
        default=50,
        help='bonus score for each second left (default: 50)',
    )

    arg_parser.add_argument(
        '--red-car-penalty',
        type=int,
        default=1000,
        help='penalty on collision with red vehicles (default: 1000; set to 0 to disable red car appearance)',
    )
    arg_parser.add_argument(
        '--yellow-car-penalty',
        type=int,
        default=500,
        help='penalty on collision with yellow vehicles (default: 500; set to 0 to disable yellow car appearance)',
    )
    arg_parser.add_argument(
        '--green-car-penalty',
        type=int,
        default=100,
        help='penalty on collision with green vehicles (default: 100; set to 0 to disable green car appearance)',
    )

    arg_parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='base seed value for configuring the initial episode of the session',
    )
    arg_parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not save any data",
    )
    arg_parser.add_argument(
        '--save-dir',
        type=str,
        default="data",
        help='the directory where all the data are saved',
    )
    args = arg_parser.parse_args()

    run_session(
        subj_id=args.subj_id,
        session_id=args.session_id,
        max_episodes=args.max_episodes,
        max_seconds=args.max_seconds,
        episode_duration_sec=args.episode_duration_sec,
        time_left_warning_threshold_sec=args.time_left_warning_threshold_sec,
        basic_score_per_grid=args.basic_score_per_grid,
        bonus_score_per_sec=args.bonus_score_per_sec,
        red_car_penalty=args.red_car_penalty,
        yellow_car_penalty=args.yellow_car_penalty,
        green_car_penalty=args.green_car_penalty,
        seed=args.seed,
        save=not args.no_save,
        base_save_dir=args.save_dir
    )
