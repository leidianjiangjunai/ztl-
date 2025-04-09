import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from utils.dali_utils import create_pipeline
from utils.gradient import DynamicGradientClipper
import torchvision.models as models

def train_step(model, inputs, labels, optimizer, scaler, clipper):
    with autocast():
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    clipper(model)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

def main():
    # 初始化模型
    model = models.resnet50(num_classes=1000).cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    clipper = DynamicGradientClipper()

    # 数据管道
    train_pipe = create_pipeline(
        data_path="../data/imagenet/train",
        batch_size=256,
        mode='train'
    )
    val_pipe = create_pipeline(
        data_path="../data/imagenet/val",
        batch_size=64,
        mode='val'
    )

    # 数据迭代器
    train_loader = DALIGenericIterator(
        train_pipe, ["images", "labels"], 
        reader_name="Reader"
    )
    val_loader = DALIGenericIterator(
        val_pipe, ["images", "labels"],
        reader_name="Reader"
    )

    # 训练循环
    for epoch in range(100):
        # 训练阶段
        model.train()
        for i, data in enumerate(train_loader):
            inputs = data[0]["images"]
            labels = data[0]["labels"].squeeze().long()
            
            loss = train_step(model, inputs, labels, optimizer, scaler, clipper)
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss:.4f}")

        # 验证阶段
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0]["images"]
                labels = data[0]["labels"].squeeze().long()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        acc = total_correct / total_samples
        print(f"Validation Accuracy: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "../models/resnet50.pth")

if __name__ == "__main__":
    main()
