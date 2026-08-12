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

def _play_session(save_dir, session_id, max_seconds, max_episodes=-1, base_seed=0):
    env = CrosswalkEnv(render_mode="human", realtime=True)
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

def run_session(subj_id=1, session_id=1, seed=0, max_episodes=-1, max_seconds=1200, save=True, base_save_dir="data"):
    if not save:
        _play_session(save_dir=None, session_id=session_id, base_seed=seed, max_episodes=max_episodes, max_seconds=max_seconds)
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

    _play_session(save_dir=save_dir, session_id=session_id, base_seed=seed, max_episodes=max_episodes, max_seconds=max_seconds)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--subj-id', type=int, default=1, help='subject ID')
    arg_parser.add_argument('--session-id', type=int, default=0, help='session ID')
    arg_parser.add_argument('--seed', type=int, default=0, help='base seed value for configuring the initial episode of the session')
    arg_parser.add_argument('--max-episodes', type=int, default=-1, help='total number of episodes')
    arg_parser.add_argument('--max-seconds', type=int, default=1200, help='ends after reaching max seconds (default: 20 min)')
    arg_parser.add_argument("--no-save", action="store_true", help="do not save any data")
    arg_parser.add_argument('--save-dir', type=str, default="data", help='the directory where all the data are saved')
    args = arg_parser.parse_args()

    run_session(
        subj_id=args.subj_id,
        session_id=args.session_id,
        seed=args.seed,
        max_episodes=args.max_episodes,
        max_seconds=args.max_seconds,
        save=not args.no_save,
        base_save_dir=args.save_dir
    )
