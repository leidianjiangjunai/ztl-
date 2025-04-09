import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配设备内存
            device_mem = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'device': device_mem, 'dtype': dtype, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'device': device_mem, 'dtype': dtype, 'shape': self.engine.get_binding_shape(binding)})
    
    def preprocess(self, image_path):
        # 与训练时相同的预处理
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img = np.array(img).astype(np.float32).transpose(2,0,1)
        img = (img - np.array([0.485*255, 0.456*255, 0.406*255])[:,None,None]) / \
              np.array([0.229*255, 0.224*255, 0.225*255])[:,None,None]
        return np.ascontiguousarray(img, dtype=np.float32)
    
    def infer(self, image_path):
        # 预处理
        input_data = self.preprocess(image_path)
        np.copyto(self.host_input, input_data.ravel())
        
        # 数据传输
        cuda.memcpy_htod(self.inputs[0]['device'], self.host_input)
        
        # 执行推理
        self.context.execute_v2(bindings=self.bindings)
        
        # 获取结果
        cuda.memcpy_dtoh(self.host_output, self.outputs[0]['device'])
        return self.host_output

if __name__ == "__main__":
    # 示例使用
    inferencer = TRTInference("../models/resnet50_int8.engine")
    result = inferencer.infer("test_image.jpg")
    print("推理结果:", np.argmax(result))
