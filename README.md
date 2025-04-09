# 1. 准备工作

## 1.1 准备数据集

本项目使用ImageNet数据集进行训练，用户需自行准备ImageNet数据集，并将其放置在`data/`目录下。数据集的目录结构应如下所示：

```
data/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...


├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   └── ...
```

## 1.2 准备预训练模型

本项目使用ResNet50作为骨干网络，用户需自行下载ResNet50的预训练模型，并将其放置在`models/`目录下。预训练模型的文件名应为`resnet50.pdparams`。

## 1.3 安装依赖

本项目依赖于PaddlePaddle、PaddleX、NVIDIA TensorRT等库，用户需自行安装这些依赖。具体安装方法请参考PaddlePaddle和PaddleX的官方文档。

# 2. 模型训练

## 2.1 训练脚本

训练脚本位于`scripts/train.py`，用户可以通过修改该脚本来调整训练参数。以下是一个示例训练命令：

```
python scripts/train.py --data-path data/ --pretrained-model models/resnet50.pdparams --save-path models/
```

## 2.2 训练参数

训练脚本支持以下参数：

- `--data-path`：数据集路径，默认为`data/`。
- `--pretrained-model`：预训练模型路径，默认为`models/resnet50.pdparams`。
- `--save-path`：模型保存路径，默认为`models/`。

# 3. 模型导出

## 3.1 ONNX模型导出

ONNX模型导出脚本位于`scripts/export_onnx.py`，用户可以通过修改该脚本来调整导出参数。以下是一个示例导出命令：

```
python scripts/export_onnx.py --model-path models/resnet50.pdparams --save-path models/resnet50.onnx
```

## 3.2 ONNX模型参数

ONNX模型导出脚本支持以下参数：

- `--model-path`：模型路径，默认为`models/resnet50.pdparams`。

- `--save-path`：模型保存路径，默认为`models/resnet50.onnx`。

# 4. TensorRT引擎构建

## 4.1 TensorRT引擎构建

TensorRT引擎构建脚本位于`scripts/build_trt.py`，用户可以通过修改该脚本来调整构建参数。以下是一个示例构建命令：

```
python scripts/build_trt.py --onnx-path models/resnet50.onnx --save-path models/resnet50.trt
```

## 4.2 TensorRT引擎参数

TensorRT引擎构建脚本支持以下参数：

- `--onnx-path`：ONNX模型路径，默认为`models/resnet50.onnx`。

- `--save-path`：引擎保存路径，默认为`models/resnet50.trt`。

# 5. 推理测试

## 5.1 推理测试脚本

推理测试脚本位于`scripts/inference.py`，用户可以通过修改该脚本来调整推理参数。以下是一个示例推理命令：

```
python scripts/inference.py --engine-path models/resnet50.trt --image-path data/val/ILSVRC2012_val_00000001.JPEG
```

## 5.2 推理参数

推理测试脚本支持以下参数：

- `--engine-path`：TensorRT引擎路径，默认为`models/resnet50.trt`。

- `--image-path`：测试图片路径，默认为`data/val/ILSVRC2012_val_00000001.JPEG`。


# 6. 总结

本项目实现了基于ResNet50的图像分类模型，并提供了从数据准备、模型训练、模型导出、TensorRT引擎构建到推理测试的全流程。用户可以根据自己的需求，修改训练参数、导出参数、构建参数和推理参数，以适应不同的应用场景。
