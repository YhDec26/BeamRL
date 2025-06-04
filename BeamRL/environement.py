import typing
from functools import partial
from typing import Optional
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from backend import TransverseTuningBaseBackend

class TransverseTuningEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 2}

    def reset(self, seed=None, options=None) -> tuple:

        self.backend.reset(seed=seed, options=options)
        if self.magnet_init_mode == "constant":
            self.backend.set_magnets(self.magnet_init_values)
        elif self.magnet_init_mode == "random":
            self.backend.set_magnets(self.observation_space["magnets"].sample())
        elif self.magnet_init_mode is None:
            pass
        else:
            raise ValueError(
                f"Invalid value '{self.magnet_init_mode} for magnet_init_mode'"
            )
        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.observation_space["target"].sample()
        else:
            raise ValueError(
                f'Invalid value "{self.target_beam_mode}" for target_beam_mode'
            )
        self.backend.update()
        self.initial_beam = self.backend.get_beam_parameters()
        self.previous_beam = self.initial_beam
        self.is_in_threshold_history = []
        self.steps_taken = 0
        observation = {
            "beam": self.initial_beam,
            "magnets": self.backend.get_magnets(),
            "target": self.target_beam,
        }
        info = {}
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        self.take_action(action)
        self.backend.update()
        current_beam = self.backend.get_beam_parameters()
        self.steps_taken += 1
        observation = {
            "beam": current_beam,
            "magnets": self.backend.get_magnets(),
            "target": self.target_beam,
        }
        cb = current_beam  
        tb = self.target_beam
        threshold = np.array(
            [
                self.target_sigma_x_threshold,
                self.target_sigma_y_threshold,
                self.target_mu_x_threshold,
                self.target_mu_y_threshold,
                self.target_mu_x_threshold,
                self.target_mu_y_threshold,
                self.target_sigma_x_threshold,
                self.target_sigma_y_threshold,
            ],
            dtype=np.double
        )
        threshold = np.nan_to_num(threshold)
        is_in_threshold = np.abs(cb - tb) < threshold
        self.is_in_threshold_history.append(is_in_threshold)
        is_stable_in_threshold = bool(
            np.array(self.is_in_threshold_history[-self.threshold_hold:]).all()
        )
        import os
        if os.getenv("DEBUG_EVAL", "0") == "1":
            print(f"[Step {self.steps_taken}] dist = {np.abs(cb - tb) * 1e3} mm, "
                  f"in_th = {is_in_threshold.tolist()}, stable = {is_stable_in_threshold}")
        done = is_stable_in_threshold and len(self.is_in_threshold_history) > 5

        time_reward = -1
        done_reward = int(done)  
        beam_reward = self.compute_beam_reward(current_beam)  

        reward = 0
        reward += self.w_beam * beam_reward  
        reward += self.w_time * time_reward  
        
        reward += self.w_sigma_x_in_threshold * is_in_threshold[0]
        reward += self.w_sigma_y_in_threshold * is_in_threshold[1]
        reward += self.w_mu_x_in_threshold * is_in_threshold[2]
        reward += self.w_mu_y_in_threshold * is_in_threshold[3]
        reward += self.w_mu_x_in_threshold * is_in_threshold[4]
        reward += self.w_mu_y_in_threshold * is_in_threshold[5]
        reward += self.w_sigma_x_in_threshold * is_in_threshold[6]
        reward += self.w_sigma_y_in_threshold * is_in_threshold[7]
        reward += self.w_done * done_reward  

        reward = float(reward)

        info = {
            "binning": self.backend.get_binning(),  
            "l1_distance": self.compute_beam_distance(current_beam, ord=1),  
            "time_reward": time_reward,  
            "is_stable_in_threshold": is_stable_in_threshold,  
        }
        info.update(self.backend.get_info())  

        self.previous_beam = current_beam  

        truncated = False

        return observation, reward, done, truncated, info

    def take_action(self, action: np.ndarray) -> None:
        """take action according to the environment's configuration"""

        if self.action_mode == "direct":
            self.backend.set_magnets(action)  
        elif self.action_mode == "delta":
            magnet_values = self.backend.get_magnets()  
            self.backend.set_magnets(magnet_values + action)  
        else:
            raise ValueError(f"Invalid value {self.action_mode} for action mode")

    def compute_beam_reward(self, current_beam: np.ndarray) -> float:
        """
        根据当前束流 current_beam 与目标束流 target_beam 之间的差异，计算本步的 beam 奖励。
        输出：float 类型 reward，越高表示 beam 越接近目标。
        """

        compute_beam_distance = partial(self.compute_beam_distance, ord=self.beam_distance_ord)

        if self.logarithmic_beam_distance:
            compute_raw_beam_distance = compute_beam_distance
            compute_beam_distance = lambda beam: np.log(  
                compute_raw_beam_distance(beam) + 1e-12
            )

        if self.reward_mode == "feedback":
            
            current_distance = compute_beam_distance(current_beam)
            beam_reward = -current_distance

        elif self.reward_mode == "differential":
            
            current_distance = compute_beam_distance(current_beam)
            previous_distance = compute_beam_distance(self.previous_beam)
            beam_reward = previous_distance - current_distance  

        
        else:
            raise ValueError(f"Invalid value '{self.reward_mode}' for reward_mode")

        if self.normalize_beam_distance:
            initial_distance = compute_beam_distance(self.initial_beam)
            eps = 1e-6  
            beam_reward /= max(initial_distance, eps)  

        
        reward = np.clip(beam_reward, -100.0, 100.0)  
        return float(reward)

    def compute_beam_distance(self, beam: np.ndarray, ord: int = 2) -> float:
        """
        计算当前 beam 参数（beam）与目标 beam（self.target_beam）之间的加权距离。
        每个 beam 参数有对应的权重（weights），使得某些参数的偏差对整体距离贡献更大/更小。

        参数：
        - beam: 当前束流参数（numpy 数组，长度 8）
        - ord: 距离范数，默认是 2 → 欧几里得距离；ord=1 则是 L1 距离（绝对值和）

        返回：
        - float，beam 与目标 beam 之间的加权距离
        """

        weights = np.array([
            self.w_sigma_x,  
            self.w_sigma_y,  
            self.w_mu_x,  
            self.w_mu_y,  
            self.w_mu_x,  
            self.w_mu_y,
            self.w_sigma_x,
            self.w_sigma_y,
        ])

        weighted_current = weights * beam
        weighted_target = weights * self.target_beam

        distance = np.linalg.norm(weighted_target - weighted_current, ord=ord)

        return float(distance)

class EATransverseTuning(TransverseTuningEnv):
    """
    Environment for positioning and focusing the beam on solenoids, Hcors and quadrupoles
    这个环境继承自 TransverseTuningEnv，专门用于在 EA 区域控制 beam：
    - 控制元件包括：Solenoids（螺线管）、Hcors（水平偏转磁铁）、Quadrupoles（四极磁铁）
    - 用于 RL 学习 beam 的自动调谐
    """
    def __init__(
            self,
            backend: TransverseTuningBaseBackend,   
            action_mode: str = "direct",            
            beam_distance_ord: int = 1,              
            logarithmic_beam_distance: bool = False, 
            magnet_init_mode: Optional[str] = None,  
            magnet_init_values: Optional[np.ndarray] = None,  

            
            max_solenoid_delta: Optional[float] = None,   
            max_quad_delta: Optional[float] = None,       
            max_steerer_delta: Optional[float] = None,    

            
            normalize_beam_distance: bool = True,   
            reward_mode: str = "feedback",      

            
            target_beam_mode: str = "random",       
            target_beam_values: Optional[np.ndarray] = None,
            target_mu_x_threshold: float = 3.3198e-6,    
            target_mu_y_threshold: float = 2.4469e-6,    
            target_sigma_x_threshold: float = 3.3198e-6, 
            target_sigma_y_threshold: float = 2.4469e-6, 
            threshold_hold: int = 1,  

            
            unidirectional_quads: bool = False,  

            
            w_beam: float = 2.0,  
            w_done: float = 5.0,  
            w_mu_x: float = 1.0, w_mu_x_in_threshold: float = 0.0,  
            w_mu_y: float = 1.0, w_mu_y_in_threshold: float = 0.0,
            w_on_screen: float = 0.0,  
            w_sigma_x: float = 1.0, w_sigma_x_in_threshold: float = 0.0,  
            w_sigma_y: float = 1.0, w_sigma_y_in_threshold: float = 0.0,  
            w_time: float = 0.0  
    ) -> None:
        self.backend = backend  

        self.action_mode = action_mode  
        self.beam_distance_ord = beam_distance_ord  
        self.logarithmic_beam_distance = logarithmic_beam_distance  

        self.magnet_init_mode = magnet_init_mode  
        self.magnet_init_values = magnet_init_values  

        
        self.max_solenoid_delta = max_solenoid_delta
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta

        
        self.normalize_beam_distance = normalize_beam_distance
        self.reward_mode = reward_mode

        
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values

        
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold

        
        self.unidirectional_quads = unidirectional_quads

        
        self.w_beam = w_beam
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        if unidirectional_quads:  
            self.magnet_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, -6.1782e-3,
                              0, 0, -6.1782e-3,
                              0, 0, -6.1782e-3,
                              0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([0.8, 0.8, 0.8, 72, 72, 6.1782e-3,
                               72, 72, 6.1782e-3,
                               72, 72, 6.1782e-3,
                               72, 72, 72, 72, 72], dtype=np.float32),
            )
        else:
            self.magnet_space = spaces.Box(
                low=np.array([0, 0, 0, -72, -72, -6.1782e-3,
                              -72, -72, -6.1782e-3,
                              -72, -72, -6.1782e-3,
                              -72, -72, -72, -72, -72], dtype=np.float32),
                high=np.array([0.8, 0.8, 0.8, 72, 72, 6.1782e-3,
                               72, 72, 6.1782e-3,
                               72, 72, 6.1782e-3,
                               72, 72, 72, 72, 72], dtype=np.float32),
            )

        
        if self.action_mode == "direct":
            self.action_space = self.magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_solenoid_delta,
                        -self.max_solenoid_delta,
                        -self.max_solenoid_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                    ],
                    dtype=np.float32
                ),
                high=np.array(
                    [
                        self.max_solenoid_delta,
                        self.max_solenoid_delta,
                        self.max_solenoid_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                    ],
                    dtype=np.float32
                )

            )
        else:
            raise ValueError(f"Invalid value '{self.action_mode}' for action_mode")

        
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([0, 0, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
                ),
                "magnets": self.magnet_space,
                "target": spaces.Box(
                    low=np.array([0, 0, -2e-3, -2e-3, -2e-3, -2e-3, 0, 0], dtype=np.float32),  
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32)
                )
            }
        )
        self.backend.setup()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        assert mode == "rgb_array" or mode == "human"

        pass

    def get_target_beam(self):
        return self.target_beam

    def get_backend_magnets(self):
        return self.backend.get_magnets()