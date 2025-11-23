# profiler/profiler.py

import time
from typing import List, Tuple, Optional

# ========== 单例元类 ==========
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ========== Profiler 类 ==========
class Profiler(metaclass=SingletonMeta):
    def __init__(self):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self._nested_level = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # GPU 内存记录: (label, timestamp, memory_in_bytes)
        self.gpu_memory_used: List[Tuple[Optional[str], float, int]] = []
        # CPU 内存（可选）
        self.cpu_memory_used: List[Tuple[Optional[str], float, int]] = []

        # 训练阶段记录: (timestamp, stage_str)
        self.stage_convert_time: List[Tuple[float, str]] = []

        # 预热结束时间（用于可视化裁剪）
        self.warmup_finish_time: Optional[float] = None

    def start(self):
        """启动 profiler（支持嵌套调用）"""
        if self._nested_level == 0:
            self.start_time = time.time()
        self._nested_level += 1

    def end(self):
        """结束 profiler"""
        self._nested_level -= 1
        if self._nested_level == 0:
            self.end_time = time.time()

    def started(self) -> bool:
        """是否处于激活状态"""
        return self._nested_level > 0

    def warmup_finish(self):
        """标记预热阶段结束"""
        self.warmup_finish_time = time.time()

    def save(self, filename: str):
        """保存所有数据到 pickle 文件"""
        import pickle
        data = {
            "gpu_memory_used": self.gpu_memory_used,
            "cpu_memory_used": self.cpu_memory_used,
            "stage_convert_time": self.stage_convert_time,
            "warmup_finish_time": self.warmup_finish_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)


# 全局唯一实例（可选，但推荐）
profiler = Profiler()