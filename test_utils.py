import unittest
import os
import shutil
import tempfile
import numpy as np
import json
from unittest.mock import MagicMock, patch

from memory_optimization.utils.config import ConfigManager, EnvironmentConfig, TrainingConfig
from memory_optimization.utils.experiment_logger import ExperimentLogger
from memory_optimization.utils.data_collector import DataCollector
from memory_optimization.utils.env_evaluator import EnvironmentEvaluator
from memory_optimization.utils.baseline_tester import BaselineTester
from memory_optimization.utils.error_handler import ErrorHandler

class TestUtils(unittest.TestCase):

    def setUp(self):
        # 创建临时目录用于测试文件生成
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def test_config_manager(self):
        """测试配置管理器"""
        config_manager = ConfigManager(config_dir=self.test_dir)
        
        # 测试默认配置
        self.assertIsInstance(config_manager.get_env_config(), EnvironmentConfig)
        self.assertIsInstance(config_manager.get_train_config(), TrainingConfig)
        
        # 测试保存和加载配置
        test_filename = os.path.join(self.test_dir, "test_config.yaml")
        config_manager.save_config(test_filename)
        self.assertTrue(os.path.exists(test_filename))
        
        # 修改配置并保存
        config_manager.update_env_config(memory_size=2048)
        config_manager.update_train_config(learning_rate=0.01)
        config_manager.save_config(test_filename)
        
        # 加载配置并验证
        new_config_manager = ConfigManager(config_dir=self.test_dir)
        new_config_manager.load_config(test_filename)
        self.assertEqual(new_config_manager.get_env_config().memory_size, 2048)
        self.assertEqual(new_config_manager.get_train_config().learning_rate, 0.01)

    def test_experiment_logger(self):
        """测试实验记录器"""
        logger = ExperimentLogger(log_dir=self.test_dir)
        self.assertTrue(os.path.exists(logger.current_experiment_dir))

        # 模拟一些指标
        metrics = {"episode_rewards": 100, "episode_lengths": 50}
        logger.log_metrics(metrics)
        self.assertEqual(logger.metrics["episode_rewards"][0], 100)

        # 模拟ConfigManager以记录配置
        mock_config_manager = MagicMock(spec=ConfigManager)
        mock_config_manager.get_env_config.return_value = EnvironmentConfig()
        mock_config_manager.get_train_config.return_value = TrainingConfig()
        logger.log_config(mock_config_manager)
        self.assertTrue(os.path.exists(os.path.join(logger.current_experiment_dir, "config.json")))

        # 测试保存指标
        logger.save_metrics()
        self.assertTrue(os.path.exists(os.path.join(logger.current_experiment_dir, "metrics.csv")))
        self.assertTrue(os.path.exists(os.path.join(logger.current_experiment_dir, "metrics.json")))

        # 测试绘图（只检查文件是否存在）
        logger.plot_metrics()
        self.assertTrue(os.path.exists(os.path.join(logger.current_experiment_dir, "training_metrics.png")))

    def test_data_collector(self):
        """测试数据收集器"""
        # 模拟一个简化的环境
        mock_env = MagicMock()
        mock_env.reset.return_value = np.array([0.1, 0.2])
        mock_env.step.side_effect = [(np.array([0.3, 0.4]), 1.0, False, {}), (np.array([0.5, 0.6]), 2.0, True, {})]
        mock_env.action_space.sample.return_value = 0 # 假设动作空间为0

        collector = DataCollector(mock_env)
        data = collector.collect_random_data(num_episodes=1)
        
        self.assertEqual(len(data["states"]), 2)
        self.assertEqual(len(data["rewards"]), 2)
        self.assertEqual(len(data["dones"]), 2)

        analysis_results = collector.analyze_data(data)
        self.assertIn("state_mean", analysis_results)
        self.assertIn("reward_mean", analysis_results)

    def test_env_evaluator(self):
        """测试环境评估器"""
        # 模拟一个简化的环境
        mock_env = MagicMock()
        mock_env.reset.return_value = np.array([0.1, 0.2])
        mock_env.step.side_effect = [ (np.array([0.3, 0.4]), 1.0, False, {}), # 模拟一步
                                     (np.array([0.5, 0.6]), 2.0, True, {}) # 模拟一步并结束
                                   ] * 50 # 模拟50个回合
        mock_env.action_space.sample.return_value = 0

        evaluator = EnvironmentEvaluator(mock_env)
        randomness_score = evaluator.evaluate_randomness(num_episodes=1) # 至少需要2个episodes才能计算标准差
        self.assertIsInstance(randomness_score, dict)
        self.assertIn('rewards_std', randomness_score)

        difficulty_metrics = evaluator.evaluate_difficulty(num_episodes=1)
        self.assertIsInstance(difficulty_metrics, dict)
        self.assertIn('average_episode_length', difficulty_metrics)

    @patch('memory_optimization.utils.baseline_tester.ExperimentLogger')
    @patch('memory_optimization.utils.baseline_tester.ConfigManager')
    def test_baseline_tester(self, MockConfigManager, MockExperimentLogger):
        """测试基线性能测试器"""
        mock_env = MagicMock()
        mock_env.reset.return_value = np.array([0.1, 0.2])
        mock_env.step.return_value = (np.array([0.3, 0.4]), 1.0, True, {"hit_rate": 0.9, "memory_usage": 0.5, "fragmentation": 0.1})
        mock_env.action_space.sample.return_value = 0

        mock_config_manager_instance = MockConfigManager.return_value
        mock_experiment_logger_instance = MockExperimentLogger.return_value

        tester = BaselineTester(mock_env, mock_config_manager_instance, log_dir=self.test_dir)
        report = tester.test_default_config(num_episodes=1)
        
        mock_experiment_logger_instance.log_config.assert_called_once()
        mock_experiment_logger_instance.log_metrics.assert_called_once()
        mock_experiment_logger_instance.save_metrics.assert_called_once()
        mock_experiment_logger_instance.plot_metrics.assert_called_once()
        self.assertIn("平均回合奖励", report)

    @patch('memory_optimization.utils.error_handler.ConfigManager')
    @patch('logging.Logger.error')
    @patch('logging.Logger.info')
    def test_error_handler(self, mock_info_logger, mock_error_logger, MockConfigManager):
        """测试错误处理器"""
        mock_config_manager_instance = MockConfigManager.return_value
        mock_config_manager_instance.get_env_config.return_value = EnvironmentConfig()
        mock_config_manager_instance.get_train_config.return_value = TrainingConfig()

        handler = ErrorHandler(mock_config_manager_instance, log_dir=self.test_dir)
        
        # 测试备份和恢复配置
        handler.backup_config()
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "config_backup.json")))

        # 模拟错误处理
        test_error = ValueError("测试错误")
        handler.handle_error(test_error)
        mock_error_logger.assert_called_with(f"发生错误: {str(test_error)}")
        mock_info_logger.assert_called_with("配置已恢复到备份状态")

        # 测试参数安全检查
        safe_params = {"learning_rate": 1e-4}
        unsafe_params = {"learning_rate": 1e-1}
        self.assertTrue(handler.check_parameter_safety(safe_params))
        self.assertFalse(handler.check_parameter_safety(unsafe_params))

if __name__ == '__main__':
    unittest.main() 