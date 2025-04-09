import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

@pipeline_def(num_threads=4, device_id=0)
def create_pipeline(data_path, batch_size=32, mode='train'):
    # 定义数据加载管道
    images, labels = fn.readers.file(
        file_root=data_path,
        random_shuffle=(mode == 'train'),
        name="Reader"
    )
    
    # 图像解码和预处理
    decoded = fn.decoders.image(images, device='mixed')
    resized = fn.resize(
        decoded,
        resize_x=256, resize_y=256,
        interp_type=types.INTERP_LINEAR
    )
    
    # 训练和验证的不同处理
    if mode == 'train':
        processed = fn.crop_mirror_normalize(
            resized,
            crop=(224, 224),
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            mirror=fn.random.coin_flip(),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT
        )
    else:
        processed = fn.crop_mirror_normalize(
            resized,
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT
        )
    
    # 标签处理
    labels = labels.gpu()
    return processed, labels
