from functools import partial
import gymnasium as gym
import wandb
import os  
import requests  
from datetime import datetime

from gymnasium.wrappers import (
    FilterObservation,   
    FlattenObservation,   
    FrameStackObservation,   
    RescaleAction,   
    TimeLimit,   
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor   
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize   
from wandb.integration.sb3 import WandbCallback

from backend import EACheetahBackend
from environement import EATransverseTuning
from utils import FilterAction, save_config

def main() -> None:
    config = {
        
        "action_mode": "delta",
        "batch_size": 512,
        "beam_distance_ord": 2,  
        "gamma": 0.99,  

        "filter_action": None,  
        "filter_observation": None,  
        "frame_stack": None,  

        "incoming_mode": "random",  
        "incoming_values": None,
        
        "learning_rate": 0.0001,
        "logarithmic_beam_distance": False,
        
        "magnet_init_mode": "random",  
        "magnet_init_values": None,  

        "max_misalignment": 5e-4,
        "max_solenoid_delta": 0.8 * 0.05,  
        "max_quad_delta": 72 * 0.1,        
        "max_steerer_delta": 6.1782e-3 * 0.1,  

        "misalignment_mode": "random",  
        "misalignment_values": None,
        "n_envs": 20,
        "n_steps": 2048,
        
        "normalize_beam_distance": True,  
        "normalize_observation": True,  
        "normalize_reward": True,

        
        "rescale_action": (-1, 1),  

        
        "reward_mode": "feedback",  
        "sb3_device": "cpu",        

        
        "target_beam_mode": "random",     
        "target_beam_values": None,       

        
        "target_mu_x_threshold": 20e-6,       
        "target_mu_y_threshold": 20e-6,       
        "target_sigma_x_threshold": 20e-6,    
        "target_sigma_y_threshold": 20e-6,    
        "threshold_hold": 3,                  

        
        "time_limit": 100,  

        
        "vec_env": "subproc",  

        
        "w_beam": 2.0,  
        "w_done": 5.0,  
        "w_mu_x": 1.0,  
        "w_mu_x_in_threshold": 0.0,  
        "w_mu_y": 1.0,
        "w_mu_y_in_threshold": 0.0,
        "w_on_screen": 0.0,  
        "w_sigma_x": 1.0,
        "w_sigma_x_in_threshold": 0.0,
        "w_sigma_y": 1.0,
        "w_sigma_y_in_threshold": 0.0,
        "w_time": 0.0,  

        
    }
    train(config)

if "WANDB_INITED" not in os.environ:
    os.environ["WANDB_INITED"] = "1"
else:
    os.environ["WANDB_MODE"] = "disabled"

os.environ['WANDB_INIT_TIMEOUT'] = '600'  

def train(config: dict) -> None:

    try:
        r = requests.get("https://api.wandb.ai", timeout=5)
        r.raise_for_status()
        print("✅ 网络正常，WandB将使用online模式")
        os.environ["WANDB_MODE"] = "online"
    except Exception:
        print("⚠️ 网络不通，WandB将切换为offline模式")
        os.environ["WANDB_MODE"] = "offline"
    
    run_mode = os.environ.get("WANDB_MODE", "unknown")
    run_name = f"{run_mode}_{config.get('project', 'clapa')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if os.environ.get("WANDB_MODE") != "disabled":
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
    
    wandb.init(
        project="clapa",                        
        name=run_name,  
        sync_tensorboard=True,                  
        monitor_gym=True,                       
        config=config,                          
        dir=".wandb",                           
        settings=wandb.Settings(init_timeout=300),  
        mode=os.environ["WANDB_MODE"],          
    )
    print(f"✅ WandB 已初始化，当前模式为: {os.environ['WANDB_MODE']}")

    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    print(f"🚀 当前 WandB 运行模式：{os.environ['WANDB_MODE']}")
    print(f"🎉 当前 WandB Run Name：{wandb.run.name}")
    
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )  
    else:
        raise ValueError(f'Invalid value \"{config["vec_env"]}"\ for dummy')
    eval_env = DummyVecEnv([partial(make_env, config, record_video=True)])

    if config["normalize_observation"] or config["normalize_reward"]:

        
        env = VecNormalize(
                env,
                norm_obs=config["normalize_observation"],
                norm_reward=config["normalize_reward"],
                gamma=config["gamma"],
        )
        
        eval_env = DummyVecEnv([partial(make_env, config, record_video=True)])
        eval_env = VecNormalize(
                eval_env,
                norm_obs=config["normalize_observation"],
                norm_reward=config["normalize_reward"],
                gamma=config["gamma"],
                training=False,        
        )
        
        eval_env.obs_rms = env.obs_rms  
        eval_env.ret_rms = env.ret_rms
        eval_env.training = False
    
    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]), 
        device=config["sb3_device"],
        gamma=config["gamma"],
        tensorboard_log=f"log/{config['run_name']}",
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    eval_callback = EvalCallback(eval_env, eval_freq=1_000, n_eval_episodes=10)
    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=4_000_00,  
        callback=[eval_callback, wandb_callback]
    )

    model.save(f"C:/Users/user/Desktop/RL×ACC/rl_for_beam_tuning-main/rl_for_beam_tuning-main/models/{wandb.run.name}/model")
    if config["normalize_observation"] or config["normalize_reward"]:
        env.save(f"C:/Users/user/Desktop/RL×ACC/rl_for_beam_tuning-main/rl_for_beam_tuning-main/models/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"C:/Users/user/Desktop/RL×ACC/rl_for_beam_tuning-main/rl_for_beam_tuning-main/models/{wandb.run.name}/config")

def make_env(config: dict, record_video: bool = False) -> gym.Env:  
    cheetah_backend = EACheetahBackend(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        max_misalignment=config["max_misalignment"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
    )  

    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode=config["action_mode"],                         
        beam_distance_ord=config["beam_distance_ord"],             
        logarithmic_beam_distance=config["logarithmic_beam_distance"],  
        magnet_init_mode=config["magnet_init_mode"],               
        magnet_init_values=config["magnet_init_values"],           
        max_solenoid_delta=config["max_solenoid_delta"],           
        max_quad_delta=config["max_quad_delta"],                   
        max_steerer_delta=config["max_steerer_delta"],             
        normalize_beam_distance=config["normalize_beam_distance"], 
        reward_mode=config["reward_mode"],                         
        target_beam_mode=config["target_beam_mode"],               
        target_beam_values=config["target_beam_values"],           
        target_mu_x_threshold=config["target_mu_x_threshold"],     
        target_mu_y_threshold=config["target_mu_y_threshold"],     
        target_sigma_x_threshold=config["target_sigma_x_threshold"],  
        target_sigma_y_threshold=config["target_sigma_y_threshold"],  
        threshold_hold=config["threshold_hold"],                   
        w_beam=config["w_beam"],                                   
        w_mu_x=config["w_mu_x"], w_mu_x_in_threshold=config["w_mu_x_in_threshold"],  
        w_mu_y=config["w_mu_y"], w_mu_y_in_threshold=config["w_mu_y_in_threshold"],
        w_sigma_x=config["w_sigma_x"], w_sigma_x_in_threshold=config["w_sigma_x_in_threshold"],
        w_sigma_y=config["w_sigma_y"], w_sigma_y_in_threshold=config["w_sigma_y_in_threshold"],
        w_on_screen=config["w_on_screen"],                         
        w_time=config["w_time"],                                   
    )

    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStackObservation(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )
    env = Monitor(env)
    return env

if __name__ == "__main__":
    main()
















