"""
处理动作执行的模块
"""

import random
import subprocess
import numpy as np

class ActionHandler:
    def __init__(self):
        # 可以定义动作空间和变化步长
        self.param_ranges = {
            'swappiness': {'min': 0, 'max': 100, 'step': 10},
            'vfs_cache_pressure': {'min': 0, 'max': 500, 'step': 25},
            'dirty_ratio': {'min': 1, 'max': 60, 'step': 5},
            'dirty_background_ratio': {'min': 1, 'max': 50, 'step': 3},
        }
    
    def execute_action(self, action):
        """执行参数调整动作"""
        # 获取当前参数
        current_params = self.get_current_parameters()
        
        # 应用动作
        for param, direction in action.items():
            if param in current_params and param in self.param_ranges:
                current_value = current_params[param]
                step = self.param_ranges[param]['step']
                
                if direction == '+':
                    new_value = min(current_value + step, self.param_ranges[param]['max'])
                elif direction == '-':
                    new_value = max(current_value - step, self.param_ranges[param]['min'])
                else:  # '0' 或其他，保持不变
                    new_value = current_value
                
                # 特殊情况：dirty_background_ratio必须小于dirty_ratio
                if param == 'dirty_background_ratio' and 'dirty_ratio' in current_params:
                    new_value = min(new_value, current_params['dirty_ratio'] - 1)
                
                # 设置新参数值
                if new_value != current_value:
                    self.set_parameter(param, new_value)
    
    def get_current_parameters(self):
        """获取当前内存子系统参数"""
        params = {}
        try:
            with open('/proc/sys/vm/swappiness', 'r') as f:
                params['swappiness'] = int(f.read().strip())
            with open('/proc/sys/vm/vfs_cache_pressure', 'r') as f:
                params['vfs_cache_pressure'] = int(f.read().strip())
            with open('/proc/sys/vm/dirty_ratio', 'r') as f:
                params['dirty_ratio'] = int(f.read().strip())
            with open('/proc/sys/vm/dirty_background_ratio', 'r') as f:
                params['dirty_background_ratio'] = int(f.read().strip())
        except Exception as e:
            print(f"Error reading parameters: {e}")
        return params
        
    def set_parameter(self, param, value):
        """设置内存子系统参数"""
        try:
            subprocess.run(['sudo', 'sysctl', f'vm.{param}={value}'], 
                         check=True, stdout=subprocess.DEVNULL)
            print(f"[Param] Set {param} to {value}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error setting parameter {param}: {e}")
            return False
    
    def get_random_action(self):
        """获取随机动作样本，用于测试和探索"""
        action = {}
        for param in self.param_ranges.keys():
            action[param] = random.choice(['+', '-', '0'])
        return action
        
    def get_action_space_size(self):
        """返回动作空间的大小"""
        # 假设你的动作空间是对每个参数进行增加/减少/不变的组合
        # 如果你有4个参数，每个参数有3种可能的动作，则动作空间大小是3^4=81
        params_count = len(self.get_parameter_list())
        actions_per_param = 3  # 增加/减少/不变
        return actions_per_param ** params_count
    
    def get_parameter_list(self):
        """返回可调整的参数列表"""
        # 例如: ['vm.swappiness', 'vm.vfs_cache_pressure', 'vm.dirty_ratio', 'vm.dirty_background_ratio']
        return list(self.get_current_parameters().keys())
