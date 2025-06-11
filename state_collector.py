"""
收集系统状态的模块
"""

import os
import time
import psutil
import numpy as np

class StateCollector:
    def __init__(self):
        pass
    
    def get_state(self):
        """获取当前系统状态，融合所有优化目标的指标"""
        state = {}
        
        # 收集页面错误信息
        self._collect_page_fault_metrics(state)
        
        # 收集内存使用信息
        self._collect_memory_metrics(state)
        
        # 收集swap使用信息
        self._collect_swap_metrics(state)
        
        # 收集缓存和slab信息
        self._collect_cache_metrics(state)
        
        # 收集系统负载信息
        self._collect_system_load(state)
        
        # 收集当前内存参数设置
        self._collect_current_parameters(state)
        
        return state
        
    def get_state_dimension(self):
        """返回状态向量的维度"""
        # 获取一个样本状态并返回其中key的数量
        sample_state = self.get_state()
        return len(sample_state)
        
    def get_state_as_array(self):
        """将状态转换为numpy数组格式"""
        state_dict = self.get_state()
        # 保持一致的顺序
        keys = sorted(state_dict.keys())
        values = [state_dict[k] for k in keys]
        return np.array(values, dtype=np.float32)
        
 
    def _collect_page_fault_metrics(self, state):
        """收集页面错误指标"""
        try:
            # 使用两次采样计算每秒错误率
            with open('/proc/vmstat', 'r') as f:
                vmstat1 = {line.split()[0]: int(line.split()[1]) for line in f 
                          if 'pgfault' in line or 'pgmajfault' in line}
            
            time.sleep(1)  # 等待1秒
            
            with open('/proc/vmstat', 'r') as f:
                vmstat2 = {line.split()[0]: int(line.split()[1]) for line in f 
                          if 'pgfault' in line or 'pgmajfault' in line}
            
            # 计算每秒错误率
            state['major_page_faults_per_sec'] = vmstat2.get('pgmajfault', 0) - vmstat1.get('pgmajfault', 0)
            state['minor_page_faults_per_sec'] = ((vmstat2.get('pgfault', 0) - vmstat1.get('pgfault', 0)) 
                                                 - state['major_page_faults_per_sec'])
        except Exception as e:
            print(f"Error collecting page fault data: {e}")
            state['major_page_faults_per_sec'] = 0
            state['minor_page_faults_per_sec'] = 0
    
    def _collect_memory_metrics(self, state):
        """收集内存使用指标"""
        try:
            vm = psutil.virtual_memory()
            state['memory_total_MB'] = vm.total / (1024 * 1024)
            state['free_memory_MB'] = vm.available / (1024 * 1024)
            state['memory_usage_percent'] = vm.percent
        except Exception as e:
            print(f"Error collecting memory metrics: {e}")
    
    def _collect_swap_metrics(self, state):
        """收集swap使用指标"""
        try:
            swap = psutil.swap_memory()
            state['swap_total_MB'] = swap.total / (1024 * 1024)
            state['swap_used_MB'] = swap.used / (1024 * 1024)
            state['swap_percent'] = swap.percent
            
            # 获取swap换入/换出速率
            with open('/proc/vmstat', 'r') as f:
                vmstat1 = {line.split()[0]: int(line.split()[1]) 
                         for line in f if 'pswpin' in line or 'pswpout' in line}
            
            time.sleep(1)  # 与前面的等待合并，实际上不需要再等1秒
            
            with open('/proc/vmstat', 'r') as f:
                vmstat2 = {line.split()[0]: int(line.split()[1]) 
                         for line in f if 'pswpin' in line or 'pswpout' in line}
            
            state['swap_in_per_sec'] = vmstat2.get('pswpin', 0) - vmstat1.get('pswpin', 0)
            state['swap_out_per_sec'] = vmstat2.get('pswpout', 0) - vmstat1.get('pswpout', 0)
        except Exception as e:
            print(f"Error collecting swap metrics: {e}")
            state['swap_in_per_sec'] = 0
            state['swap_out_per_sec'] = 0
    
    def _collect_cache_metrics(self, state):
        """收集缓存和slab指标"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    if ':' in line:
                        key, value = line.split(':')
                        if len(value.split()) > 1:
                            value_number, unit = value.strip().split()
                            if unit == 'kB':
                                meminfo[key.strip()] = float(value_number) / 1024  # 转换为MB
                        else:
                            meminfo[key.strip()] = float(value.strip())
            
            state['slab_memory_MB'] = meminfo.get('Slab', 0)
            state['cached_memory_MB'] = meminfo.get('Cached', 0)
            state['buffers_memory_MB'] = meminfo.get('Buffers', 0)
            state['active_memory_MB'] = meminfo.get('Active', 0)
            state['inactive_memory_MB'] = meminfo.get('Inactive', 0)
        except Exception as e:
            print(f"Error collecting cache metrics: {e}")
    
    def _collect_system_load(self, state):
        """收集系统负载信息"""
        try:
            state['system_loadavg_1min'] = os.getloadavg()[0]
            state['system_loadavg_5min'] = os.getloadavg()[1]
            state['cpu_usage_percent'] = psutil.cpu_percent(interval=0.5)
            if hasattr(psutil, 'cpu_times_percent'):
                cpu_times = psutil.cpu_times_percent(interval=0.5)
                state['cpu_iowait_percent'] = getattr(cpu_times, 'iowait', 0)
        except Exception as e:
            print(f"Error collecting system load: {e}")
    
    def _collect_current_parameters(self, state):
        """收集当前内存参数设置"""
        try:
            with open('/proc/sys/vm/swappiness', 'r') as f:
                state['current_swappiness'] = int(f.read().strip())
            with open('/proc/sys/vm/vfs_cache_pressure', 'r') as f:
                state['current_cache_pressure'] = int(f.read().strip())
            with open('/proc/sys/vm/dirty_ratio', 'r') as f:
                state['current_dirty_ratio'] = int(f.read().strip())
            with open('/proc/sys/vm/dirty_background_ratio', 'r') as f:
                state['current_dirty_bg_ratio'] = int(f.read().strip())
        except Exception as e:
            print(f"Error reading memory parameters: {e}")
            
