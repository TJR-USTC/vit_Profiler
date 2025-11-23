# profile_visualizer.py
import argparse
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def load_profile_data(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)

def visualize_memory(
    filename: str,
    memory_type: str = "GPU",
    rm_warmup: bool = False,
    output: Optional[str] = None,
):
    """
    可视化 GPU 或 CPU 内存使用情况
    """
    data = load_profile_data(filename)
    
    if memory_type.upper() == "GPU":
        raw_memory = data.get("gpu_memory_used", [])
        ylabel = "GPU Memory (MB)"
        color = "tab:red"
    elif memory_type.upper() == "CPU":
        raw_memory = data.get("cpu_memory_used", [])
        ylabel = "CPU Memory (MB)"
        color = "tab:orange"
    else:
        raise ValueError("memory_type must be 'GPU' or 'CPU'")
    
    if not raw_memory:
        print(f"No {memory_type} memory data found in {filename}")
        return

    # 提取时间戳和内存值
    timestamps = [t for (_, t, _) in raw_memory]
    memory_mb = [m / (1024 ** 2) for (_, _, m) in raw_memory]

    # 确定起始时间（是否移除 warmup）
    start_time = data.get("warmup_finish_time") if rm_warmup else data.get("start_time")
    if start_time is None:
        start_time = min(timestamps) if timestamps else 0

    # 转换为相对时间（秒）
    relative_time = [t - start_time for t in timestamps]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(relative_time, memory_mb, color=color, linewidth=1.5, label=f"{memory_type} Memory")

    # 获取训练阶段信息（用于着色背景）
    stage_times = data.get("stage_convert_time", [])
    if stage_times:
        stage_colors = {"FWD": "green", "BWD": "blue", "ADAM": "purple"}
        prev_t = 0.0
        for i, (t_abs, stage) in enumerate(stage_times):
            t_rel = t_abs - start_time
            if t_rel < 0:
                continue
            # 上一阶段从 prev_t 到 t_rel
            stage_name = stage_times[i - 1][1] if i > 0 else "IDLE"
            color_bg = stage_colors.get(stage_name, "white")
            plt.axvspan(prev_t, t_rel, color=color_bg, alpha=0.1)
            prev_t = t_rel
        # 最后一段到结束
        if relative_time:
            last_stage = stage_times[-1][1] if stage_times else "IDLE"
            color_bg = stage_colors.get(last_stage, "white")
            plt.axvspan(prev_t, relative_time[-1], color=color_bg, alpha=0.1)

    plt.xlabel("Time (seconds)")
    plt.ylabel(ylabel)
    plt.title(f"{memory_type} Memory Usage Over Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Figure saved to {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize PatrickStar-style profiler output.")
    parser.add_argument("--filename", type=str, required=True, help="Path to .pkl profile file")
    parser.add_argument("--memory_type", type=str, default="GPU", choices=["GPU", "CPU"],
                        help="Memory type to visualize (default: GPU)")
    parser.add_argument("--rm_warmup", action="store_true",
                        help="Remove warmup phase from the plot")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to file (e.g., mem.png). If not set, show interactively.")

    args = parser.parse_args()
    visualize_memory(
        filename=args.filename,
        memory_type=args.memory_type,
        rm_warmup=args.rm_warmup,
        output=args.output
    )

if __name__ == "__main__":
    main()