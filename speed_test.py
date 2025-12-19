import torch
import time
import argparse
import sys
import os

# 假设你的模型定义在 model.py 中，类名为 LiteAMSNet
# 请根据实际情况修改这里的 import
try:
    from model import LiteAMSNet 
except ImportError:
    # 为了演示，如果找不到 model，我定义一个假的
    import torch.nn as nn
    class LiteAMSNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(3, 64, 3)
        def forward(self, x):
            return self.conv(x)

def measure_speed(checkpoint_path, input_shape=(1, 3, 200), device='cuda'):
    # 1. 加载模型
    model = LiteAMSNet()
    # 如果你的模型需要参数初始化，请在这里添加
    
    # 尝试加载权重
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 处理可能的 DataParallel 'module.' 前缀
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # 2. 准备 Dummy Input
    dummy_input = torch.randn(input_shape).to(device)

    # 3. Warm up (预热 GPU)
    print("🔥 Warming up GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # 4. 正式测试
    iterations = 1000
    print(f"⏱️ Running {iterations} iterations for latency measurement...")
    
    # 同步 CUDA 时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()

    # 5. 计算指标
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000 # ms
    fps = iterations / total_time

    print("-" * 30)
    print(f"✅ Results for {checkpoint_path}")
    print(f"   Avg Latency: {avg_latency:.4f} ms")
    print(f"   Throughput : {fps:.2f} FPS")
    print("-" * 30)
    
    return avg_latency, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    # 这里的 input_shape 需要根据你真实数据的 shape 修改！
    # 比如 (Batch=1, Channels=3, Length=你的窗口长度)
    measure_speed(args.checkpoint, input_shape=(1, 3, 128))