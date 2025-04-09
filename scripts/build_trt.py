import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.calibrator import EntropyCalibrator

def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # 量化配置
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = EntropyCalibrator(
        data_dir="../data/calib_data",
        cache_file="../models/calibration.cache"
    )
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # 网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open("../models/resnet50.onnx", "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX解析错误:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 优化配置
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input", 
        min=(1, 3, 224, 224),
        opt=(32, 3, 224, 224), 
        max=(64, 3, 224, 224)
    )
    config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    with open("../models/resnet50_int8.engine", "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    build_engine()
