import torch

class DynamicGradientClipper:
    def __init__(self, init_clip=0.1, max_clip=1.0, factor=0.1):
        self.clip_value = init_clip
        self.max_clip = max_clip
        self.min_clip = init_clip
        self.factor = factor

    def __call__(self, model):
        # 计算梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 动态调整裁剪阈值
        if total_norm > self.clip_value:
            self.clip_value = min(self.clip_value * (1 + self.factor), self.max_clip)
        else:
            self.clip_value = max(self.clip_value * (1 - self.factor), self.min_clip)

        # 执行梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
