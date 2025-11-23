# profiler/memory_monitor.py

import torch
import psutil
import os

def get_sys_memory_used(device: torch.device) -> int:
    """
    获取设备当前已分配内存（bytes）
    """
    if device.type == "cuda":
        # 注意：使用 memory_allocated 而非 reserved
        return torch.cuda.memory_allocated(device)
    elif device.type == "cpu":
        # 返回当前进程的物理内存使用量
        pid = os.getpid()
        process = psutil.Process(pid)
        return process.memory_info().rss
    else:
        raise ValueError(f"Unsupported device type: {device.type}")