from functools import partial
import os                                     
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import (
    FlattenObservation,
    TimeLimit,
    RecordVideo,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backend import EACheetahBackend
from environment import EATransverseTuning
from utils import load_config                 

MODEL_PATH = r"C:\Users\user\Desktop\RL×ACC\rl_for_beam_tuning-main\rl_for_beam_tuning-main\models\offline_clapa_20250601_183837"
NUM_EPISODES = 10            
RECORD_VIDEO  = False        

def main() -> None:
    
    config = load_config(os.path.join(MODEL_PATH, "config"))       
    
    for k in ["incoming_values", "magnet_init_values", "target_beam_values"]:
        if isinstance(config.get(k), list):
            config[k] = np.array(config[k], dtype=np.float32)       

    config["target_mu_x_threshold"] = 100e-6
    config["target_mu_y_threshold"] = 100e-6
    config["target_sigma_x_threshold"] = 100e-6
    config["target_sigma_y_threshold"] = 100e-6
    
    env = create_eval_env(config)
    model, env = load_model_and_env(MODEL_PATH, env)
    evaluate_model(model, env, num_episodes=NUM_EPISODES)

def create_eval_env(config: dict) -> gym.Env:
    """根据训练时的配置创建评估环境"""
    env = DummyVecEnv([partial(make_env, config)])

    vec_path = os.path.join(MODEL_PATH, "vec_normalize.pkl")        
    if os.path.exists(vec_path) and (
        config.get("normalize_observation") or config.get("normalize_reward")
    ):
        env = VecNormalize.load(vec_path, env)                      
        env.training = False
        env.norm_reward = False

    if RECORD_VIDEO:
        env = RecordVideo(
            env,
            video_folder=os.path.join(MODEL_PATH, "videos"),
            episode_trigger=lambda x: x % 2 == 0,
        )

    return env

def load_model_and_env(model_path: str, env: gym.Env):
    """载入 PPO 模型并绑定到同一环境"""
    model = PPO.load(
        os.path.join(model_path, "model"),  
        env=env,
        device=torch.device("cpu"),
        print_system_info=True,
    )
    return model, env

def evaluate_model(model, env, num_episodes=10):
    success_count, total_rewards, beam_errors = 0, [], []

    for ep in range(num_episodes):
        obs = env.reset()
        done, ep_reward, ep_steps = False, 0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            ep_reward += reward[0]
            ep_steps  += 1
            done_flag  = done[0] if isinstance(done, (list, np.ndarray)) else done  
            if done_flag:
                beam_params = obs[0][:8]
                target      = env.env_method("get_target_beam")[0]
                print("\n=== [DEBUG EVAL] ===")
                print("obs[0][:8]:", beam_params)
                print("target:    ", target)

                beam_errors.append(np.linalg.norm(beam_params - target))
                
                if info[0].get("is_stable_in_threshold", False):
                    success_count += 1
                break

        total_rewards.append(ep_reward)
        print(f"\nEpisode {ep+1}/{num_episodes}")
        print(f"Steps: {ep_steps} | Reward: {ep_reward:.2f}")
        print(f"Beam Error: {beam_errors[-1]:.4e}")
        print("Final Beam Parameters:", obs[0][:8].round(6))  

    print("\n=== Evaluation Summary ===")
    print(f"Success Rate  : {success_count / num_episodes:.1%}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(
        f"Mean Beam Error: {np.mean(beam_errors):.4e} ± {np.std(beam_errors):.4e}"
    )

def make_env(config: dict) -> gym.Env:
    backend = EACheetahBackend(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        misalignment_mode=config.get("misalignment_mode", "random"),
    )

    env = EATransverseTuning(
        backend=backend,
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=config["target_beam_values"],
        max_solenoid_delta=config["max_solenoid_delta"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
    )

    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=config.get("time_limit", 300))
    return Monitor(env)

if __name__ == "__main__":
    main()
