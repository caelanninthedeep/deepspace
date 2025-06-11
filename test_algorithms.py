import unittest
import torch
import numpy as np
from memory_optimization.algorithms.ppo import PPO
from memory_optimization.algorithms.buffer import ReplayBuffer
from memory_optimization.algorithms.networks import PolicyNetwork, ValueNetwork

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.state_dim = 12  # 根据实际状态维度调整
        self.action_dim = 9  # 根据实际动作维度调整
        self.ppo = PPO(self.state_dim, self.action_dim)
        self.buffer = ReplayBuffer(capacity=100)

    def test_policy_network(self):
        """测试策略网络"""
        policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        state = torch.randn(1, self.state_dim) # 批次大小为1
        action_probs = policy_net(state)
        self.assertEqual(action_probs.shape, (1, self.action_dim))
        self.assertTrue(torch.all(action_probs >= 0)) # 动作概率应非负

    def test_value_network(self):
        """测试价值网络"""
        value_net = ValueNetwork(self.state_dim)
        state = torch.randn(1, self.state_dim)
        value = value_net(state)
        self.assertEqual(value.shape, (1, 1))

    def test_replay_buffer(self):
        """测试回放缓冲区"""
        state = np.random.rand(self.state_dim)
        action = np.random.randint(self.action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(self.state_dim)
        done = False
        
        self.buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer.buffer), 1)
        
        # 填充缓冲区到容量
        for _ in range(self.buffer.capacity - 1):
            self.buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer.buffer), self.buffer.capacity)
        
        # 测试采样
        batch_size = 32
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        self.assertEqual(len(states), batch_size)
        self.assertIsInstance(states[0], np.ndarray)

    def test_ppo_select_action(self):
        """测试PPO的动作选择"""
        state = np.random.rand(self.state_dim)
        action = self.ppo.select_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_ppo_compute_returns(self):
        """测试PPO的回报计算"""
        rewards = [1, 1, 1, 1]
        dones = [0, 0, 0, 1] # 最后一个是终止
        returns = self.ppo.compute_returns(rewards, dones)
        self.assertIsInstance(returns, torch.Tensor)
        self.assertEqual(returns.shape, torch.Size([len(rewards)]))
        
        # 简单验证回报值
        # R_n = r_n
        # R_{n-1} = r_{n-1} + gamma * R_n
        # 假设gamma=0.99
        expected_returns = [
            1 + 0.99 * (1 + 0.99 * (1 + 0.99 * 1)), # roughly
            1 + 0.99 * (1 + 0.99 * 1),
            1 + 0.99 * 1,
            1
        ]
        # 检查最后一个回报是否正确
        self.assertAlmostEqual(returns[-1].item(), rewards[-1], places=5)
        # 检查倒数第二个回报
        self.assertAlmostEqual(returns[-2].item(), rewards[-2] + self.ppo.gamma * rewards[-1], places=5)

    def test_ppo_update(self):
        """测试PPO的更新逻辑（功能性检查）"""
        # 准备一些模拟数据
        num_steps = 64
        states = [np.random.rand(self.state_dim) for _ in range(num_steps)]
        actions = [np.random.randint(self.action_dim) for _ in range(num_steps)]
        rewards = [np.random.rand() for _ in range(num_steps)]
        next_states = [np.random.rand(self.state_dim) for _ in range(num_steps)]
        dones = [False] * (num_steps - 1) + [True]
        
        # 在更新前保存网络参数
        old_policy_params = {k: v.clone() for k, v in self.ppo.policy_net.named_parameters()}
        old_value_params = {k: v.clone() for k, v in self.ppo.value_net.named_parameters()}
        
        # 执行更新
        self.ppo.update(states, actions, rewards, next_states, dones)
        
        # 检查参数是否发生变化（说明更新逻辑被执行）
        policy_changed = False
        for name, param in self.ppo.policy_net.named_parameters():
            if not torch.equal(old_policy_params[name], param):
                policy_changed = True
                break
        self.assertTrue(policy_changed, "策略网络的参数未更新")
        
        value_changed = False
        for name, param in self.ppo.value_net.named_parameters():
            if not torch.equal(old_value_params[name], param):
                value_changed = True
                break
        self.assertTrue(value_changed, "价值网络的参数未更新")

if __name__ == '__main__':
    unittest.main() 