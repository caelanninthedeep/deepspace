"""
内存子系统优化强化学习环境的核心类
"""

import time
import random
from collections import deque
import numpy as np
import gym
from gym import spaces

from .state_collector import StateCollector
from .action_handler import ActionHandler
from .reward_calculator import RewardCalculator
from memory_optimization.workloads.memory_workloads import MemoryWorkload
from memory_optimization.workloads.io_workloads import IOWorkload
from memory_optimization.workloads.mixed_workloads import MixedWorkload
from memory_optimization.utils.safety_checks import SafetyChecker

class MemoryOptimizationEnv(gym.Env):
    """Linux内存子系统优化的强化学习环境"""
    
    metadata = {'render.modes': ['human']}

    def __init__(self, params={}):
        super(MemoryOptimizationEnv, self).__init__()
        
        self.params = params
        
        # 初始化组件
        self.state_collector = StateCollector()
        self.action_handler = ActionHandler()
        self.reward_calculator = RewardCalculator()
        self.safety_checker = SafetyChecker()
        
        # 工作负载实例
        self.memory_workload = MemoryWorkload()
        self.io_workload = IOWorkload()
        self.mixed_workload = MixedWorkload()
        
        # 初始化历史记录
        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.runtime_history = deque(maxlen=10)
        
        # 初始状态
        self.state = None
        self.previous_workload_runtime = None
        
        # 记录初始参数值以便恢复
        self.initial_params = self.action_handler.get_current_parameters()
        
        # 设置动作空间
        # ActionHandler已经有get_action_space_size方法
        self.action_space = spaces.Discrete(self.action_handler.get_action_space_size())
        
        # 设置观察空间
        # 获取一个样例状态来确定维度
        sample_state = self.state_collector.get_state()
        state_dim = len(sample_state)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
    def reset(self):
        """重置环境状态"""
        print("[RL Env] Resetting environment...")
        
        # 恢复初始参数
        if hasattr(self, 'initial_params'):
            for param, value in self.initial_params.items():
                self.action_handler.set_parameter(param, value)
                
        # 清空历史记录
        self.action_history.clear()
        self.state_history.clear()
        self.reward_history.clear()
        self.runtime_history.clear()
        
        # 初始状态
        self.state = self.state_collector.get_state()
        self.state_history.append(self.state)
        
        # 运行初始工作负载以建立基准
        self.run_test_workload()
        self.previous_workload_runtime = self.last_workload_runtime
        
        # 更新状态以包含工作负载运行时间
        self.state = self.state_collector.get_state()
        
        # 将状态字典转换为numpy数组
        state_array = self._state_dict_to_array(self.state)
        
        return state_array
        
    def step(self, action):
        """执行一步动作"""
        # 安全检查
        if not self.safety_checker.is_safe_to_proceed(self.state):
            print("[SAFETY] Unsafe system state detected. Skipping this step.")
            return self._state_dict_to_array(self.state), -100, False, {"skipped": True}
        
        # 将整数动作转换为字典形式
        action_dict = self._convert_action_to_dict(action)
        
        # 记录动作
        self.action_history.append(action)
        
        # 执行参数调整
        self.action_handler.execute_action(action_dict)
        
        # 运行测试工作负载
        self.run_test_workload()
        
        # 获取新状态
        new_state = self.state_collector.get_state()
        self.state = new_state
        self.state_history.append(new_state)
        
        # 计算奖励
        reward = self.reward_calculator.calculate_reward(new_state, self.previous_workload_runtime, 
                                                     self.last_workload_runtime)
        self.reward_history.append(reward)
        
        # 更新前一次工作负载运行时间
        self.previous_workload_runtime = self.last_workload_runtime
        
        # 判断是否结束
        done = False
        # 可以根据实际需求设置结束条件
        
        info = {
            'last_workload_type': getattr(self, 'last_workload_type', None),
            'last_workload_runtime': getattr(self, 'last_workload_runtime', None),
        }
        
        # 将状态字典转换为numpy数组
        state_array = self._state_dict_to_array(new_state)
        
        return state_array, reward, done, info
    
    def _convert_action_to_dict(self, action):
        """将整数动作转换为字典形式"""
        params = self.action_handler.get_parameter_list()
        actions_per_param = 3  # 增加/减少/不变
        
        action_dict = {}
        action_value = action
        
        # 从右到左解码每个参数的动作
        for param in reversed(params):
            action_for_param = action_value % actions_per_param
            action_value //= actions_per_param
            
            if action_for_param == 0:
                action_dict[param] = '+'
            elif action_for_param == 1:
                action_dict[param] = '-'
            else:  # action_for_param == 2
                action_dict[param] = '0'
                
        return action_dict
    
    def _state_dict_to_array(self, state_dict):
        """将状态字典转换为numpy数组"""
        # 使用固定的顺序，确保一致性
        keys = sorted(state_dict.keys())
        values = [float(state_dict[k]) for k in keys]  # 确保所有值为浮点数
        return np.array(values, dtype=np.float32)
    
    def run_test_workload(self):
        """运行测试负载并测量性能"""
        # 随机选择一种工作负载类型
        workload_types = [
            'memory_intensive',
            'io_intensive',
            'mixed_workload',
            'file_cache_intensive'
        ]
        workload_type = random.choice(workload_types)
        
        # 记录开始时间
        start_time = time.time()
        
        # 根据选择的类型运行对应工作负载
        if workload_type == 'memory_intensive':
            self.memory_workload.run_memory_intensive()
        elif workload_type == 'io_intensive':
            self.io_workload.run_io_intensive()
        elif workload_type == 'mixed_workload':
            self.mixed_workload.run_mixed_workload()
        elif workload_type == 'file_cache_intensive':
            self.io_workload.run_file_cache_intensive()
        
        # 记录结束时间和运行时长
        end_time = time.time()
        self.last_workload_runtime = end_time - start_time
        self.last_workload_type = workload_type
        
        print(f"[Workload] {workload_type} completed in {self.last_workload_runtime:.2f} seconds")
    
    def get_action_space_sample(self):
        """获取随机动作样本，用于测试和探索"""
        return self.action_space.sample()  # 使用Gym标准的sample方法
    
    def render(self, mode='human'):
        """渲染环境状态（可选）"""
        if mode == 'human':
            # 打印当前状态和参数
            current_params = self.action_handler.get_current_parameters()
            print("\n=== Current Environment State ===")
            print(f"State: {self.state}")
            print(f"Parameters: {current_params}")
            if hasattr(self, 'last_workload_runtime'):
                print(f"Last workload ({self.last_workload_type}): {self.last_workload_runtime:.2f}s")
            if len(self.reward_history) > 0:
                print(f"Last reward: {self.reward_history[-1]}")
            print("===============================\n")
    
    def close(self):
        """关闭环境，恢复初始参数"""
        print("[RL Env] Closing environment and restoring parameters...")
        if hasattr(self, 'initial_params'):
            for param, value in self.initial_params.items():
                self.action_handler.set_parameter(param, value)
