import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import subprocess
import time
import os
import logging
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import json

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('memory_optimizer.log')  # 输出到文件
    ]
)
logger = logging.getLogger('MemoryOptimizer')

class SystemStateProcessor:
    def __init__(self, csv_path, sequence_length=5, validation_split=0.2):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'mem_usage',
            'swap_usage',
            'avail_ratio',
            'pgfault_diff',
            'pgmajfault_diff',
            'cpu_usage',
            'load_avg_1m',
            'context_switches',
            'disk_read',
            'disk_write',
            'net_recv_rate',
            'net_sent_rate'
        ]
        self.df = self._load_and_preprocess(csv_path)
        self.train_size = int(len(self.df) * (1 - validation_split))
        self.validation_df = self.df[self.train_size:]
        self.train_df = self.df[:self.train_size]
        
    def _load_and_preprocess(self, csv_path):
        # 读取CSV数据
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 计算差分指标
        df['pgfault_diff'] = df['pgfault'].diff().fillna(0)
        df['pgmajfault_diff'] = df['pgmajfault'].diff().fillna(0)
        
        # 计算内存使用率
        df['mem_usage'] = 1 - (df['mem_available'] / df['mem_total'])
        
        # 计算交换空间使用率
        df['swap_usage'] = 1 - (df['swap_free'] / df['swap_total'])
        
        # 计算可用内存比例
        df['avail_ratio'] = df['mem_available'] / df['mem_total']
        
        # 添加更多特征
        df['memory_pressure'] = df['pgfault_diff'] / df['mem_total']
        df['swap_pressure'] = df['swap_usage'] * df['pgmajfault_diff']
        df['io_pressure'] = (df['disk_read'] + df['disk_write']) / df['mem_total']
        
        # 选择关键特征
        features = df[self.feature_columns]
        
        # 训练归一化器 - 使用 .values 避免特征名警告
        self.scaler.fit(features.values)
        logger.info("Scaler fitted successfully")
        
        return df
    
    def get_state_vector(self, index, is_validation=False):
        # 选择数据集
        df = self.validation_df if is_validation else self.train_df
        
        # 获取序列数据
        start_idx = max(0, index - self.sequence_length + 1)
        sequence = df.loc[start_idx:index, self.feature_columns].values # 获取 NumPy 数组
        
        # 填充序列
        if len(sequence) < self.sequence_length:
            padding = np.tile(sequence[0], (self.sequence_length - len(sequence), 1)) # 填充 NumPy 数组
            sequence = np.vstack([padding, sequence]) # 垂直堆叠
        
        # 归一化
        normalized = self.scaler.transform(sequence)
        return normalized # 返回形状为 (sequence_length, input_size) 的 NumPy 数组
    
    def get_validation_states(self):
        """获取验证集状态"""
        states = []
        for i in range(len(self.validation_df)):
            state = self.get_state_vector(i, is_validation=True)
            states.append(state)
        return np.array(states)
    
    def find_next_state(self, current_state_vector, action): # 接收状态向量作为输入
        """根据当前状态和动作找到最接近的下一个状态"""
        # 将动作参数转换为特征向量
        action_features = self._action_to_features(action)

        # 计算当前状态与所有可能的下一个状态的相似度
        similarities = []
        # 确保 train_df 有足够的数据进行查找
        if len(self.train_df) < 2:
             logger.warning("train_df has less than 2 rows, cannot find next state.")
             # 返回当前状态作为下一个状态的模拟
             # 这里需要返回字典格式
             return self.train_df.iloc[-1].to_dict() if len(self.train_df) > 0 else {}


        for i in range(len(self.train_df) - 1):
            # 获取下一个状态的特征向量 (2D numpy array)
            next_state_features_seq = self.get_state_vector(i+1, is_validation=False)
            # 使用序列的最后一个时间步的状态特征进行相似度计算
            next_state_features = next_state_features_seq[-1, :]

            # 计算状态相似度（使用欧氏距离）
            # current_state_vector 是 (seq_len, input_size)，next_state_features 是 (input_size,)
            # 我们需要比较当前状态序列的最后一个状态与下一个状态
            state_similarity = np.linalg.norm(current_state_vector[-1, :] - next_state_features)

            # 计算动作相似度 - 使用 train_df 中当前时间步的特征作为该时间步的动作特征模拟
            # 这里的逻辑是将历史数据中的系统指标视为一种"动作特征"，来匹配当前的动作
            # 这是一种简化的模拟，可能与实际的参数-指标关系不完全一致
            # 这里的 historical_action_features 应该也来自 train_df.iloc[i]
            historical_features = self.train_df.iloc[i][self.feature_columns].values # 获取当前时间步的特征
            # 对历史特征进行归一化以便与 action_features 比较
            normalized_historical_features = self.scaler.transform(historical_features.reshape(1, -1)).flatten()


            action_similarity = np.linalg.norm(action_features - normalized_historical_features)
            # 综合相似度
            similarity = state_similarity + action_similarity
            similarities.append((i+1, similarity))

        # 选择最相似的下一个状态
        # 确保 similarities 不为空
        if not similarities:
             logger.warning("No similar states found.")
             # 返回当前状态作为下一个状态的模拟
             # 这里需要返回字典格式
             return self.train_df.iloc[-1].to_dict() if len(self.train_df) > 0 else {}

        next_idx = min(similarities, key=lambda x: x[1])[0]
        next_state_row = self.train_df.iloc[next_idx]

        # 返回状态字典
        # 确保从 row 转换到字典时包含所有需要的键，即使这些键不在原始 CSV 中
        next_state_dict = next_state_row.to_dict()
        # 添加 VM 参数的默认值，如果它们不在原始数据中
        for param, details in VM_PARAMETERS.items():
            if param not in next_state_dict:
                 next_state_dict[param] = details['default']

        return next_state_dict


    def _action_to_features(self, action):
        """将动作参数转换为特征向量"""
        features = np.zeros(len(self.feature_columns))
        # 根据动作参数更新特征
        # 这里的映射是一种简化的表示，将 VM 参数值映射到特征向量的特定位置
        # 这种映射的合理性取决于参数与这些特征的相关性
        try:
            if 'swappiness' in action and action['swappiness'] is not None:
                features[self.feature_columns.index('mem_usage')] = float(action['swappiness']) / 100  # 假设 swappiness 与 mem_usage 相关
            if 'min_free_kbytes' in action and action['min_free_kbytes'] is not None:
                # 使用 avail_ratio 特征
                # 注意：这里将 min_free_kbytes 映射到 avail_ratio。
                # avail_ratio 的范围是 [0, 1]。需要确保映射函数也输出在这个范围内。
                # 409600 是一个假设的最大值，可能需要根据实际数据调整。
                max_min_free = 409600 # 假设的最大值
                min_free_kbytes_val = float(action['min_free_kbytes'])
                # 将 min_free_kbytes 映射到可用比例，更高的 min_free_kbytes 对应更高的可用比例
                mapped_avail_ratio = min_free_kbytes_val / max_min_free # 简单的线性映射，范围 [0, 1]
                features[self.feature_columns.index('avail_ratio')] = mapped_avail_ratio

            if 'vfs_cache_pressure' in action and action['vfs_cache_pressure'] is not None:
                 # vfs_cache_pressure 范围 [50, 200]。映射到 pgfault_diff (差分值，无固定范围)
                 # 简单的线性映射到 [0, 1]
                 min_val, max_val = VM_PARAMETERS['vfs_cache_pressure']['range']
                 pressure_val = float(action['vfs_cache_pressure'])
                 mapped_pgfault_diff = (pressure_val - min_val) / (max_val - min_val) # 映射到 [0, 1]
                 # 这里将映射值赋给 pgfault_diff 特征
                 features[self.feature_columns.index('pgfault_diff')] = mapped_pgfault_diff

            if 'dirty_ratio' in action and action['dirty_ratio'] is not None:
                 # dirty_ratio 范围 [10, 40]，映射到 disk_write。
                 # 简单的线性映射到 [0, 1]
                 min_val, max_val = VM_PARAMETERS['dirty_ratio']['range']
                 dirty_ratio_val = float(action['dirty_ratio'])
                 mapped_disk_write = (dirty_ratio_val - min_val) / (max_val - min_val) # 映射到 [0, 1]
                 features[self.feature_columns.index('disk_write')] = mapped_disk_write

            if 'dirty_background_ratio' in action and action['dirty_background_ratio'] is not None:
                 # dirty_background_ratio 范围 [5, 20]，也映射到 disk_write。
                 # 将其影响加到 disk_write 特征上。
                 min_val, max_val = VM_PARAMETERS['dirty_background_ratio']['range']
                 bg_dirty_val = float(action['dirty_background_ratio'])
                 mapped_disk_write_bg = (bg_dirty_val - min_val) / (max_val - min_val) # 映射到 [0, 1]
                 features[self.feature_columns.index('disk_write')] += mapped_disk_write_bg # 累加影响


        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error converting action to features: {e}")
            # 如果转换出错，返回零向量或者默认特征向量
            return np.zeros(len(self.feature_columns))

        # 对特征进行归一化，使用训练时的 scaler
        # 注意：这里是对根据动作模拟出的"特征"进行归一化，
        # 这与对实际系统状态特征进行归一化是不同的概念。
        # 这种做法的合理性值得商榷，它假设了动作可以直接映射到归一化后的系统特征空间。
        # 更严谨的做法是重新考虑如何将动作融入状态表示或奖励计算。
        # 为了与现有 find_next_state 逻辑兼容，暂时保留此处的归一化。
        try:
             # 这里的 features 是形状为 (input_size,) 的 numpy 数组
             # scaler.transform 期望输入是 (n_samples, n_features)
             normalized_features = self.scaler.transform(features.reshape(1, -1))
             return normalized_features.flatten() # 返回形状为 (input_size,) 的一维数组
        except Exception as e:
             logger.warning(f"Error normalizing action features: {e}")
             return features # 归一化失败，返回原始特征


# 定义可调参数及其范围
VM_PARAMETERS = {
    'swappiness': {
        'range': (0, 100),
        'steps': [0, 20, 40, 60, 80, 100],
        'default': 60,
        'sysctl_path': '/proc/sys/vm/swappiness'
    },
    'min_free_kbytes': {
        'range': (51200, 409600),  # 50MB - 400MB
        'steps': [51200, 102400, 204800, 307200, 409600],
        'default': 204800,
        'sysctl_path': '/proc/sys/vm/min_free_kbytes'
    },
    'vfs_cache_pressure': {
        'range': (50, 200),
        'steps': [50, 100, 150, 200],
        'default': 100,
        'sysctl_path': '/proc/sys/vm/vfs_cache_pressure'
    },
    'dirty_ratio': {
        'range': (10, 40),
        'steps': [10, 20, 30, 40],
        'default': 20,
        'sysctl_path': '/proc/sys/vm/dirty_ratio'
    },
    'dirty_background_ratio': {
        'range': (5, 20),
        'steps': [5, 10, 15, 20],
        'default': 10,
        'sysctl_path': '/proc/sys/vm/dirty_background_ratio'
    }
}

# 生成动作编号表
ACTION_TABLE = []
param_keys = list(VM_PARAMETERS.keys())

# 创建笛卡尔积动作空间
for swappiness in VM_PARAMETERS['swappiness']['steps']:
    for min_free in VM_PARAMETERS['min_free_kbytes']['steps']:
        for cache_pressure in VM_PARAMETERS['vfs_cache_pressure']['steps']:
            for dirty_ratio in VM_PARAMETERS['dirty_ratio']['steps']:
                for bg_dirty in VM_PARAMETERS['dirty_background_ratio']['steps']:
                    action = {
                        'id': len(ACTION_TABLE),
                        'params': {
                            'swappiness': swappiness,
                            'min_free_kbytes': min_free,
                            'vfs_cache_pressure': cache_pressure,
                            'dirty_ratio': dirty_ratio,
                            'dirty_background_ratio': bg_dirty
                        }
                    }
                    ACTION_TABLE.append(action)

logger.info(f"Total actions generated: {len(ACTION_TABLE)}")

def verify_setting(param, expected):
    """验证参数是否设置成功"""
    try:
        with open(VM_PARAMETERS[param]['sysctl_path'], "r") as f:
            actual = int(f.read().strip())
            if actual != expected:
                logger.warning(f"Failed to set {param}: expected {expected}, got {actual}")
                return False
        return True
    except Exception as e:
        logger.error(f"Verification failed for {param}: {str(e)}")
        return False

# 注释掉 apply_action 函数的实际实现，因为我们目前是模拟环境
# def apply_action(action_params):
#    """应用连续动作到系统参数 - 模拟"""
#    logger.info(f"Simulating applying action: {action_params}")
#    # 在模拟环境中，我们不真正修改系统参数
#    # 这里可以添加一些逻辑来模拟动作执行的结果，例如延迟
#    # time.sleep(random.uniform(1, 3)) # 模拟一个随机延迟
#    # 总是返回成功，因为是模拟
#    return True

# def verify_setting(param, expected):
#     """验证参数是否设置成功 - 模拟"""
#     # 在模拟环境中，我们不真正验证系统参数设置
#     return True

class LSTMMemoryOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, num_layers=2):
        super(LSTMMemoryOptimizer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        return self.actor(last_hidden), self.critic(last_hidden)

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.model = LSTMMemoryOptimizer(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.memory = []
        self.gamma = 0.99
        self.lambda_gae = 0.95
        logger.info(f"PPO Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def get_action(self, state):
        """获取动作和值函数估计"""
        # state 现在是形状为 (sequence_length, input_size) 的 numpy 数组
        # 需要添加 batch 维度 (大小为 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # 添加 batch 维度
        with torch.no_grad():
            action_probs, value = self.model(state)
        return action_probs, value # action_probs: (1, action_size), value: (1, 1)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验 - state 和 next_state 现在是形状为 (sequence_length, input_size) 的 numpy 数组"""
        # action 是原始动作向量 (action_size,) numpy 数组
        self.memory.append((state, action, reward, next_state, done))
    
    def compute_gae(self, rewards, values, next_value, dones):
        """计算广义优势估计"""
        # rewards, values, next_value, dones 应该是 PyTorch 张量
        advantages = []
        gae = 0

        # 确保 next_value 是一个张量，即使它是一个标量
        if isinstance(next_value, (int, float)):
             next_value_tensor = torch.tensor([next_value], dtype=torch.float32)
        else:
             next_value_tensor = next_value

        # 反向遍历计算优势
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 最后一个时间步，下一个值是传入的 next_value
                next_value_t = next_value_tensor
            else:
                # 非最后一个时间步，下一个值是 values 中的下一个元素
                next_value_t = values[t + 1]

            # 计算时使用张量操作
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            # 将计算得到的优势张量添加到列表中
            advantages.insert(0, gae)

        # 将优势列表堆叠成一个张量
        # 使用 torch.stack 而不是 torch.tensor 来堆叠张量列表
        # .detach() 是可选的，因为我们是在 torch.no_grad() 中计算优势，但为了安全可以加上
        return torch.stack(advantages).detach()
    
    def update(self, batch_size=64):
        """更新策略"""
        if len(self.memory) < batch_size:
            return
        
        # 准备数据
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 计算优势
        with torch.no_grad():
            values = self.model(states)[1].squeeze()
            if next_states.shape[0] > 0:
                 next_value = self.model(next_states[-1:])[1].squeeze()
                 if next_value.ndim == 0:
                     next_value = next_value.unsqueeze(0)
            else:
                 next_value = torch.tensor([0.0], dtype=torch.float32)

        if len(values) != len(dones):
             logger.error(f"Mismatch in lengths: values={len(values)}, dones={len(dones)}")
             self.memory = []
             return

        # 调用 compute_gae 时传入 PyTorch 张量
        # ensure values and dones are 1D tensors if batch_size is 1
        values_1d = values.flatten() if values.ndim > 0 else values.unsqueeze(0)
        dones_1d = dones.flatten() if dones.ndim > 0 else dones.unsqueeze(0)

        advantages = self.compute_gae(rewards, values_1d, next_value, dones_1d) # 传入张量

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算策略损失
        for _ in range(10):  # 多次更新
            # 对于连续动作空间，这里不使用 gather
            # batch_actions 已经是原始动作向量

            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx + batch_size]
                batch_actions = actions[idx:idx + batch_size]
                batch_advantages = advantages[idx:idx + batch_size]

                # 计算新的动作概率和值
                # self.model(batch_states) 期望输入 (batch_size, seq_len, input_size)
                new_action_probs, new_values = self.model(batch_states) # new_action_probs: (batch_size, action_size), new_values: (batch_size, 1)

                # 计算值函数损失
                # new_values 是 (batch_size, 1)，rewards[idx:idx + batch_size] 是 (batch_size,)
                # 需要确保形状一致
                value_loss = self.value_coef * nn.MSELoss()(new_values.squeeze(), rewards[idx:idx + batch_size])

                # 计算熵损失 (可选，通常用于连续 PPO 鼓励探索)
                # 如果 new_action_probs 是均值，熵依赖于标准差。
                # 鉴于当前模型结构，无法准确计算熵。
                entropy_loss = torch.tensor(0.0) # 暂时设为0

                # PPO 策略损失部分需要实现
                # 计算旧动作的对数概率 (需要从 memory 中存储的 old_log_probs 获取)
                # 计算新动作的对数概率
                # 计算 ratio = exp(new_log_probs - old_log_probs)
                # 计算 clipped_ratio
                # policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # 总损失
                # loss = policy_loss + value_loss + entropy_loss # PPO 总损失
                loss = value_loss + entropy_loss # 暂时只包含值损失和熵损失，策略损失未实现


                # 优化
                self.optimizer.zero_grad()
                # loss.backward() # 在 PPO 中，通常策略损失和值损失的 backward 是分开的
                # loss.backward() # 先对总损失进行一次 backward
                value_loss.backward() # 先对值损失进行 backward (确保值网络更新)
                # TODO: 实现策略损失并进行 backward
                self.optimizer.step()

        # 清空记忆
        self.memory = []
    
    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")

    def save_config(self, path):
        config = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.model.lstm.hidden_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae
        }
        with open(path, 'w') as f:
            json.dump(config, f)

    def log_metrics(self, episode, total_reward, avg_reward, loss):
        self.metrics['episode'].append(episode)
        self.metrics['total_reward'].append(total_reward)
        self.metrics['avg_reward'].append(avg_reward)
        self.metrics['loss'].append(loss)

class RewardCalculator:
    def __init__(self):
        # 初始化权重
        self.weights = {
            'maj_faults': 0.5,
            'swap_ratio': 0.3,
            'avail_ratio': 0.2,
            'cpu_overload': 0.2,
            'param_change': 0.3,
            'oscillation': 0.5,
            'perf_improvement': 0.1
        }
        self.emergency_mode = False
        # 动作历史现在存储动作字典
        self.action_history = []
        self.tunable_params = [
            'swappiness',
            'min_free_kbytes',
            'vfs_cache_pressure',
            'dirty_ratio',
            'dirty_background_ratio'
        ]

    def calculate_reward(self, current_state, previous_state, previous_action):
        """
        计算当前状态下的奖励值
        :param current_state: 当前系统状态字典 (从CSV数据获取的指标)
        :param previous_state: 前一个系统状态字典 (从CSV数据获取的指标)
        :param previous_action: 前一个执行的动作字典
        :return: 奖励值
        """
        # 1. 基础性能奖励
        reward = 0

        # 1.1 惩罚主要页错误（每次错误扣0.5分）
        # 确保键存在并且是数值类型
        maj_faults = 0
        # 检查 current_state 和 previous_state 是否有 pgmajfault 键
        if previous_state and 'pgmajfault' in current_state and 'pgmajfault' in previous_state:
             # 确保值是数值类型
             try:
                 maj_faults = float(current_state['pgmajfault']) - float(previous_state['pgmajfault'])
                 maj_faults = max(0, maj_faults) # 页错误差值不应该为负
             except (ValueError, TypeError):
                 logger.warning("pgmajfault is not a valid number in state.")
                 maj_faults = 0 # 如果不是有效数字，则视为0变化

        reward -= self.weights['maj_faults'] * maj_faults

        # 1.2 惩罚交换空间使用（按使用比例扣分）
        swap_ratio = 0
        if 'swap_total' in current_state and 'swap_free' in current_state and current_state['swap_total'] is not None and current_state['swap_total'] > 0:
            try:
                swap_used = float(current_state['swap_total']) - float(current_state['swap_free'])
                swap_ratio = swap_used / float(current_state['swap_total'])
                swap_ratio = max(0, min(1, swap_ratio)) # 确保比例在 [0, 1] 之间
            except (ValueError, TypeError):
                logger.warning("swap_total or swap_free is not a valid number in current_state.")
                swap_ratio = 0 # 如果不是有效数字，则视为0

        # 在 SystemStateProcessor 中添加了 avail_ratio 计算，这里也使用 avail_ratio
        # 1.3 奖励可用内存（按可用比例加分）
        avail_ratio = 0
        if 'avail_ratio' in current_state and current_state['avail_ratio'] is not None: # 使用 avail_ratio 键
             try:
                 avail_ratio = float(current_state['avail_ratio'])
                 avail_ratio = max(0, min(1, avail_ratio)) # 确保比例在 [0, 1] 之间
             except (ValueError, TypeError):
                 logger.warning("avail_ratio is not a valid number in current_state.")
                 avail_ratio = 0 # 如果不是有效数字，则视为0

        reward += self.weights['avail_ratio'] * avail_ratio

        # 1.4 惩罚高负载（超过CPU核心数时扣分）
        cpu_cores = os.cpu_count() if os.cpu_count() is not None else 1 # 确保cpu_cores不为None
        overload = 0
        if 'load_avg_1m' in current_state and current_state['load_avg_1m'] is not None: # 统一为 load_avg_1m
            try:
                load_avg = float(current_state['load_avg_1m'])
                if load_avg > cpu_cores:
                    overload = load_avg - cpu_cores
            except (ValueError, TypeError):
                logger.warning("load_avg_1m is not a valid number in current_state.")
                overload = 0 # 如果不是有效数字，则视为0

        reward -= self.weights['cpu_overload'] * overload

        # 2. 稳定性奖励
        # 2.1 参数变动惩罚（使用平滑函数）- 基于动作变化
        param_change = 0
        # 计算当前动作和上一个动作之间的参数变化
        # action_history[-1] 是当前动作，previous_action 是上一个动作
        if previous_action and previous_action != {} and self.action_history:
             current_action_dict = self.action_history[-1]
             for param in self.tunable_params:
                  # 确保参数存在于两个动作字典中
                  if param in current_action_dict and param in previous_action:
                     try:
                         param_change += abs(float(current_action_dict[param]) - float(previous_action[param]))
                     except (ValueError, TypeError):
                         logger.warning(f"Parameter {param} in action is not a valid number.")


        if param_change > 0: # 只有参数发生变化才惩罚
             # 使用平滑函数，惩罚随变化量增大
             # 归一化 param_change 到一个合理的范围
             # 假设总变化量最大为所有参数范围之和
             # 修正了这里的访问方式
             max_total_change = sum(r['range'][1] - r['range'][0] for r in VM_PARAMETERS.values() if 'range' in r and isinstance(r['range'], tuple) and len(r['range']) == 2)
             normalized_param_change = param_change / max_total_change if max_total_change > 0 else 0

             # 使用平方惩罚，变化越大惩罚越多
             punish_factor = normalized_param_change ** 2

             reward -= self.weights['param_change'] * punish_factor


        # 2.2 震荡抑制（避免反复调整） - 基于动作历史
        oscillation_penalty = 0
        # 检查动作历史是否有至少三个动作，以便判断最近两次变化方向
        if len(self.action_history) >= 3:
             last_action = self.action_history[-1]
             second_last_action = self.action_history[-2]
             third_last_action = self.action_history[-3]

             for param in self.tunable_params:
                  # 确保参数存在于最近三个动作字典中
                  if param in last_action and param in second_last_action and param in third_last_action:
                     try:
                         # 计算最近两次的变化方向
                         change1 = float(last_action[param]) - float(second_last_action[param])
                         change2 = float(second_last_action[param]) - float(third_last_action[param])

                         # 如果两次变化方向相反（一个正一个负），则存在震荡
                         if (change1 > 0 and change2 < 0) or (change1 < 0 and change2 > 0):
                              oscillation_penalty = self.weights['oscillation']
                              break # 只要有一个参数震荡就惩罚
                     except (ValueError, TypeError):
                         logger.warning(f"Parameter {param} in action history is not a valid number.")

        reward -= oscillation_penalty


        # 3. 紧急情况处理
        # 3.1 内存危机模式
        if avail_ratio < 0.05:  # 可用内存<5%
            self.emergency_mode = True
            # 页错误惩罚加倍
            reward -= self.weights['maj_faults'] * maj_faults # 已经在前面乘以了基础权重，这里不再乘以2，而是直接加大权重
        else:
            self.emergency_mode = False

        # 3.2 交换空间枯竭
        swap_free_ratio = 1
        if 'swap_total' in current_state and 'swap_free' in current_state and current_state['swap_total'] is not None and current_state['swap_total'] > 0:
            try:
                 swap_free_ratio = float(current_state['swap_free']) / float(current_state['swap_total'])
                 swap_free_ratio = max(0, min(1, swap_free_ratio))
            except (ValueError, TypeError):
                logger.warning("swap_total or swap_free is not a valid number in current_state for free ratio.")
                swap_free_ratio = 1 # 如果不是有效数字，则视为完全空闲

        if swap_free_ratio < 0.1:
            # 交换使用惩罚加倍
            reward -= self.weights['swap_ratio'] * swap_ratio # 已经在前面乘以了基础权重，这里不再乘以2，而是直接加大权重

        # 4. 性能趋势奖励
        perf_improvement = 0
        # 检查 previous_state 是否存在，并且需要的键都在
        if previous_state and 'MemAvailable' in current_state and 'MemAvailable' in previous_state and 'mem_total' in current_state and current_state['mem_total'] is not None and current_state['mem_total'] > 0:
            try:
                # 计算内存改善
                mem_available_current = float(current_state['MemAvailable'])
                mem_available_previous = float(previous_state['MemAvailable'])
                mem_total = float(current_state['mem_total'])

                perf_improvement = (mem_available_previous - mem_available_current) / mem_total

                if perf_improvement > 0:
                    reward += self.weights['perf_improvement'] * perf_improvement
            except (ValueError, TypeError):
                logger.warning("MemAvailable or mem_total is not a valid number for performance improvement calculation.")


        # 5. 记录奖励计算过程
        logger.debug(
            f"Reward components: "
            f"maj_faults={maj_faults:.2f}, " # 格式化输出，避免科学计数法
            f"swap_ratio={swap_ratio:.3f}, "
            f"avail_ratio={avail_ratio:.3f}, "
            f"param_change={param_change:.2f}, " # 记录参数变化总和
            f"oscillation_penalty={oscillation_penalty:.2f}, " # 记录震荡惩罚
            f"emergency_mode={self.emergency_mode}, "
            f"perf_improvement={perf_improvement:.3f}"
        )

        # 6. 限制奖励范围
        return max(min(reward, 1.0), -1.0)

    def update_weights(self, system_state):
        """
        根据系统状态动态调整权重
        """
        # 根据内存使用情况调整权重
        avail_ratio = 1
        if 'avail_ratio' in system_state and system_state['avail_ratio'] is not None: # 使用 avail_ratio 键
            try:
                avail_ratio = float(system_state['avail_ratio'])
            except (ValueError, TypeError):
                logger.warning("avail_ratio is not a valid number for weight update.")
                avail_ratio = 1

        if avail_ratio < 0.05:
            self.weights['maj_faults'] = 1.0 # 危机时加大惩罚权重
            self.weights['swap_ratio'] = 0.6 # 危机时加大惩罚权重
        else:
            self.weights['maj_faults'] = 0.5
            self.weights['swap_ratio'] = 0.3

        # 根据系统负载调整权重
        cpu_cores = os.cpu_count() if os.cpu_count() is not None else 1
        load_avg = system_state.get('load_avg_1m') # 统一为 load_avg_1m
        if load_avg is not None:
            try:
                load_avg = float(load_avg)
                if load_avg > cpu_cores * 1.5:
                    self.weights['cpu_overload'] = 0.4 # 高负载时加大惩罚权重
                else:
                    self.weights['cpu_overload'] = 0.2
            except (ValueError, TypeError):
                logger.warning("load_avg_1m is not a valid number for weight update.")
        else:
             self.weights['cpu_overload'] = 0.2 # 如果负载数据无效，使用默认权重


        logger.debug(f"Updated weights: {self.weights}")

    def add_action(self, action):
        """
        记录动作历史
        """
        # 记录动作字典
        self.action_history.append(action)
        if len(self.action_history) > 10:  # 只保留最近10个动作
            self.action_history.pop(0)

class ContinuousActionSpace:
    def __init__(self):
        self.param_ranges = {
            'swappiness': (0, 100),
            'min_free_kbytes': (51200, 409600),
            'vfs_cache_pressure': (50, 200),
            'dirty_ratio': (10, 40),
            'dirty_background_ratio': (5, 20)
        }
        self.param_names = list(self.param_ranges.keys())
    
    def scale_action(self, action):
        """将[-1,1]范围的动作映射到实际参数范围"""
        scaled_params = {}
        for i, (param, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            scaled_params[param] = int(min_val + (action[i] + 1) * (max_val - min_val) / 2)
        return scaled_params
    
    def get_action_size(self):
        return len(self.param_ranges)

class AdaptiveWaitTime:
    def __init__(self, min_wait=5, max_wait=20):
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.current_wait = min_wait
        
    def update(self, system_response):
        """根据系统响应调整等待时间"""
        if system_response > self.current_wait:
            self.current_wait = min(self.current_wait * 1.2, self.max_wait)
        else:
            self.current_wait = max(self.current_wait * 0.8, self.min_wait)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float('-inf')
        self.counter = 0
        
    def __call__(self, reward):
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class ModelEvaluator:
    def __init__(self, agent, state_processor, reward_calculator):
        self.agent = agent
        self.state_processor = state_processor
        self.reward_calculator = reward_calculator
        self.best_reward = float('-inf')
        self.best_model_state = None
        self.action_space = None
    
    def evaluate(self):
        """评估模型在验证集上的表现"""
        validation_states_vector = self.state_processor.get_validation_states()
        total_reward = 0

        # 在评估循环开始前，保存当前的 action_history
        original_action_history = list(self.reward_calculator.action_history)
        # 清空 action_history，避免评估过程影响训练历史
        self.reward_calculator.action_history = []

        # 获取验证集对应的状态字典列表
        initial_state_idx = self.state_processor.train_size
        if initial_state_idx >= len(self.state_processor.df):
             logger.warning("Validation set is empty or too small.")
             # 评估结束后，恢复原始的 action_history
             self.reward_calculator.action_history = original_action_history
             return 0

        validation_states_dict = [self.state_processor.df.iloc[i].to_dict() for i in range(initial_state_idx, len(self.state_processor.df))]

        logger.info("Starting model evaluation on validation set.")

        # 评估循环：对于验证集中的每个状态，获取动作并计算奖励
        # 在评估时，我们模拟单个时间步的奖励，不进行连续交互。
        # previous_action 设为默认动作，用于评估的第一个 step。
        simulated_previous_action = {param: VM_PARAMETERS[param]['default'] for param in VM_PARAMETERS}


        for i in range(len(validation_states_vector)):
            state_vector = validation_states_vector[i]
            current_state_dict = validation_states_dict[i]

            # 获取动作 (原始动作向量)
            action_probs, _ = self.agent.get_action(state_vector) # action_probs 是 tensor (1, action_size)
            raw_action_vector = action_probs[0].numpy() # 原始动作向量 (action_size,) numpy

            # 将原始动作缩放到实际参数字典，用于传递给 RewardCalculator 的 calculate_reward (模拟)
            if self.action_space is None:
                 logger.error("action_space is not set in ModelEvaluator.")
                 # 评估结束后，恢复原始的 action_history
                 self.reward_calculator.action_history = original_action_history
                 return float('-inf')

            # 模拟的当前动作 (scaled)
            simulated_current_action_scaled = self.action_space.scale_action(raw_action_vector)


            # 在评估时，将模拟的当前动作添加到 action_history（临时），以便 calculate_reward 计算 param_change 和 oscillation
            self.reward_calculator.add_action(simulated_current_action_scaled)


            # 计算奖励
            # 模拟执行动作后的下一个状态
            # 注意：find_next_state 需要 state_vector 和 action (scaled)
            simulated_next_state_dict = self.state_processor.find_next_state(state_vector, simulated_current_action_scaled)

            # 在评估阶段，计算从当前状态 (current_state_dict) 采取 simulated_current_action_scaled 动作后，
            # 到达模拟的下一个状态 (simulated_next_state_dict) 所获得的奖励。
            # 前一个状态是 current_state_dict，前一个动作是 simulated_previous_action。
            reward = self.reward_calculator.calculate_reward(
                current_state=simulated_next_state_dict,
                previous_state=current_state_dict,
                previous_action=simulated_previous_action
            )

            total_reward += reward

            # 更新模拟的前一个动作，用于下一个验证状态的奖励计算
            simulated_previous_action = simulated_current_action_scaled


        # 评估结束后，恢复原始的 action_history
        self.reward_calculator.action_history = original_action_history

        avg_reward = total_reward / (len(validation_states_vector) if validation_states_vector else 1) # 避免除以零

        # 保存最佳模型
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_model_state = self.agent.model.state_dict()
            logger.info(f"New best model found with reward: {avg_reward:.2f}")

        return avg_reward
    
    def save_best_model(self, path):
        """保存最佳模型"""
        if self.best_model_state is not None:
            torch.save(self.best_model_state, path)
            logger.info(f"Best model saved to {path}")

    def evaluate_model(self):
        metrics = {
            'memory_usage': [],
            'swap_usage': [],
            'page_faults': [],
            'response_time': []
        }
        # 收集评估指标
        return metrics
    
def main():
    logger.info("Starting memory optimization training")
    
    # 初始化
    # 这里的路径需要根据你的实际文件位置进行修改
    # 例如：'/home/hirene/Desktop/OS比赛/数据收集/memory_metrics.csv'
    # 或者使用相对于当前工作目录的路径：'数据收集/memory_metrics.csv'
    # 请根据你的实际情况修改下面这一行
    # 根据你之前的报错，'/home/asdf/数据收集/数据收集/memory_metrics.csv' 可能是正确的路径
    csv_path = '/home/hirene/Desktop/OS比赛/数据收集/memory_metrics.csv'
    state_processor = SystemStateProcessor(csv_path)
    state_size = len(state_processor.feature_columns)
    action_space = ContinuousActionSpace()
    action_size = action_space.get_action_size()
    
    # 初始化智能体和辅助类
    agent = PPOAgent(state_size, action_size)
    reward_calculator = RewardCalculator()
    wait_time = AdaptiveWaitTime()
    early_stopping = EarlyStopping()
    evaluator = ModelEvaluator(agent, state_processor, reward_calculator)
    evaluator.action_space = action_space
    
    batch_size = 64
    episodes = 100
    max_steps = 100 # 每个 episode 的最大步数
    eval_interval = 5 # 每隔多少个 episode 进行一次评估
    
    # 训练循环
    for episode in range(episodes):
        logger.info(f"Starting episode {episode+1}/{episodes}")
        
        # 重置环境 (从数据集的开头开始一个新的 episode)
        state_idx = 0
        # 确保数据集足够长，可以开始新的 episode
        if state_idx >= len(state_processor.train_df):
             logger.warning(f"Dataset too short to start episode {episode+1}. Stopping training.")
             break

        # 获取初始状态向量 (形状为 (seq_len, input_size))
        state_vector = state_processor.get_state_vector(state_idx)
        # 获取初始状态字典，作为第一个 step 的 previous_state
        state = state_processor.train_df.iloc[state_idx].to_dict()
        total_reward = 0
        episode_rewards = []
        # 初始化 previous_action 为默认参数字典，作为第一个 step 的前一个动作
        previous_action = {param: VM_PARAMETERS[param]['default'] for param in VM_PARAMETERS}
        # 清空每个 episode 的 action_history，确保震荡惩罚是针对当前 episode 内的行为
        reward_calculator.action_history = []


        for step in range(max_steps):
            logger.debug(f"Episode {episode+1}, Step {step+1}")

            # 获取动作 (原始动作向量)
            # agent.get_action 期望输入形状为 (seq_len, input_size) 的 numpy 数组
            action_probs, value = agent.get_action(state_vector) # action_probs 是 tensor (1, action_size), value 是 tensor (1, 1)
            raw_action_vector = action_probs[0].numpy() # 原始动作向量 (action_size,) numpy

            # 将原始动作缩放到实际参数字典
            action = action_space.scale_action(raw_action_vector)

            # 执行动作 (模拟)
            # 在这个模拟环境中，我们不真正执行动作，而是假设动作成功并影响到下一个状态的选择
            # success = apply_action(action) # 在真实环境中需要执行并检查结果
            # 这里假设动作总是"成功"的，因为是基于历史数据查找下一个状态

            # 将缩放后的当前动作添加到动作历史
            reward_calculator.add_action(action)

            # 获取新状态 (基于历史数据)
            # find_next_state 期望输入当前状态向量 (seq_len, input_size) 和当前动作字典
            next_state_dict = state_processor.find_next_state(state_vector, action)

            # 获取下一个时间步的状态向量 (形状为 (seq_len, input_size))
            # 在模拟环境中，下一个状态向量应该对应数据集中的下一个时间步的数据
            next_state_idx = state_idx + 1

            # 判断 episode 是否结束 (达到最大步数或数据结束)
            done = (step == max_steps - 1) or (next_state_idx >= len(state_processor.train_df))

            # 如果数据结束，下一个状态向量和状态字典都使用当前最后一个有效状态的数据
            if next_state_idx >= len(state_processor.train_df):
                next_state_vector = state_vector # 数据结束，状态向量不再更新
                next_state_dict = state # 数据结束，状态字典不再更新
                logger.debug("End of dataset reached.")
            else:
                 next_state_vector = state_processor.get_state_vector(next_state_idx)
                 # next_state_dict 已经在上面通过 find_next_state 获取

            # 计算奖励
            # current_state 是观察到的下一个状态字典 (next_state_dict)
            # previous_state 是当前步之前的状态字典 (state)
            # previous_action 是上一个循环迭代中生成的动作字典
            reward = reward_calculator.calculate_reward(
                current_state=next_state_dict,
                previous_state=state,
                previous_action=previous_action
            )

            total_reward += reward
            episode_rewards.append(reward)

            # 存储经验
            # 经验是 (state, action, reward, next_state, done)
            # state 是当前循环迭代开始时的状态向量 (state_vector - (seq_len, input_size))
            # action 是当前循环迭代中智能体选择的原始动作向量 (raw_action_vector - (action_size,))
            # reward 是观察到 next_state 后计算的奖励
            # next_state 是下一个时间步的状态向量 (next_state_vector - (seq_len, input_size))
            # done 表示 episode 是否结束
            agent.remember(state_vector, raw_action_vector, reward, next_state_vector, done)

            # 更新状态和索引
            state = next_state_dict # 当前状态字典更新为下一个状态字典
            state_vector = next_state_vector # 当前状态向量更新为下一个状态向量
            previous_action = action # 当前缩放后的动作字典成为下一个 step 的前一个动作
            state_idx = next_state_idx # 更新数据索引

            # 更新模型
            # 在每个 step 或累积一定经验后更新
            # 只有当 memory 大小达到 batch_size 时才更新
            if len(agent.memory) >= batch_size:
                logger.debug("Updating agent policy.")
                agent.update(batch_size)
                # 更新后清空 memory 在 agent.update 中完成

            # 如果 episode 结束，跳出 step 循环
            if done:
                logger.info(f"Episode {episode+1} completed at step {step+1}")
                break
        

        # 计算平均奖励
        # 避免除以零
        avg_reward = total_reward / (len(episode_rewards) if episode_rewards else 1)
        logger.info(f"Episode {episode+1}/{episodes} completed - "
                   f"Total Reward: {total_reward:.2f}, "
                   f"Average Reward: {avg_reward:.2f}")

        # 定期评估
        if (episode + 1) % eval_interval == 0:
            # 在评估前，记录当前的 agent 模型状态，评估后恢复
            current_model_state = agent.model.state_dict()
            # 保存 agent memory 和 reward_calculator action_history，评估后恢复
            original_agent_memory = list(agent.memory)
            original_rc_action_history = list(reward_calculator.action_history)
            agent.memory = [] # 清空 agent memory 进行评估
            reward_calculator.action_history = [] # 清空 rc action_history 进行评估

            try:
                validation_reward = evaluator.evaluate()
                logger.info(f"Validation reward: {validation_reward:.2f}")
            except Exception as e:
                 logger.error(f"Error during evaluation: {e}")
                 validation_reward = float('-inf') # 评估失败，给予负奖励

            # 评估后，恢复 agent 模型状态、memory 和 reward_calculator action_history
            agent.model.load_state_dict(current_model_state)
            agent.memory = original_agent_memory
            reward_calculator.action_history = original_rc_action_history


        # 早停检查
        # 使用 episode 的平均奖励进行早停判断
        if early_stopping(avg_reward):
            logger.info("Early stopping triggered")
            break

    # 保存最佳模型
    evaluator.save_best_model('memory_optimizer_best.pth')
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
