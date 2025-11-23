# profiler/torch_profiler_hook.py

import torch
import time
from .profiler import profiler
from .memory_monitor import get_sys_memory_used


def _record_mem_stats():
    """记录当前 GPU 和 CPU 内存"""
    if not profiler.started():
        return

    # 记录 GPU 显存（假设使用 cuda:0）
    try:
        gpu_mem = get_sys_memory_used(torch.device("cuda"))
        profiler.gpu_memory_used.append((None, time.time(), gpu_mem))
    except Exception:
        pass  # 无 GPU 时跳过

    # 记录 CPU 内存
    try:
        cpu_mem = get_sys_memory_used(torch.device("cpu"))
        profiler.cpu_memory_used.append((None, time.time(), cpu_mem))
    except Exception:
        pass


def _pre_forward_hook(module, input):
    _record_mem_stats()

def _post_forward_hook(module, input, output):
    _record_mem_stats()

def _pre_backward_hook(module, input, output):
    _record_mem_stats()

def _post_backward_hook(module, input):
    _record_mem_stats()


def register_torch_profiler_hook(model: torch.nn.Module):
    """
    为模型递归注册内存采样钩子
    """
    for name, child in model.named_children():
        register_torch_profiler_hook(child)

    # 只对有参数的模块注册钩子（避免冗余）
    if len(list(model.parameters(recurse=False))) > 0:
        model.register_forward_pre_hook(_pre_forward_hook)
        model.register_forward_hook(_post_forward_hook)
        # 为输出张量注册反向钩子
        from torch.autograd import Function

        class PreBackwardFunc(Function):
            @staticmethod
            def forward(ctx, x):
                return x.detach()
            @staticmethod
            def backward(ctx, grad_output):
                _pre_backward_hook(None, None, None)
                return grad_output

        class PostBackwardFunc(Function):
            @staticmethod
            def forward(ctx, x):
                return x.detach()
            @staticmethod
            def backward(ctx, grad_output):
                _post_backward_hook(None, None)
                return grad_output

        def wrap_output_with_backward_hook(output):
            if isinstance(output, torch.Tensor):
                if output.requires_grad:
                    return PostBackwardFunc.apply(PreBackwardFunc.apply(output))
                return output
            elif isinstance(output, (tuple, list)):
                return type(output)(wrap_output_with_backward_hook(x) for x in output)
            else:
                return output

        original_forward = model.forward
        def new_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            return wrap_output_with_backward_hook(output)
        model.forward = new_forward