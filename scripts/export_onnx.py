import torch
import torch.onnx
from torch import nn
import torchvision.models as models

def export_onnx_model():
    # 加载训练好的模型
    model = models.resnet50()
    model.load_state_dict(torch.load("../models/resnet50.pth"))
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    
    # 导出设置
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "../models/resnet50.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13
    )

if __name__ == "__main__":
    export_onnx_model()
