import os
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file, batch_size=32):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        
        # 加载校准数据
        self.image_list = []
        with open(os.path.join(data_dir, "calib_list.txt")) as f:
            for line in f:
                self.image_list.append(os.path.join(data_dir, "images", line.strip()))
        
        # 分配内存
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 224 * 224 * 4)
        
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img = np.array(img).astype(np.float32).transpose(2,0,1)
        img = (img - np.array([0.485*255, 0.456*255, 0.406*255])[:,None,None]) / \
              np.array([0.229*255, 0.224*255, 0.225*255])[:,None,None]
        return np.ascontiguousarray(img, dtype=np.float32)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_list):
            return None
        
        batch_images = []
        for _ in range(self.batch_size):
            img = self.preprocess_image(self.image_list[self.current_index])
            batch_images.append(img)
            self.current_index += 1
        
        np_batch = np.stack(batch_images, axis=0).ravel()
        cuda.memcpy_htod(self.device_input, np_batch)
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
