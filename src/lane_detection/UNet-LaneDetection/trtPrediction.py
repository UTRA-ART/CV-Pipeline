import numpy as np
import torch
import cv2

import trt_helper
import time


engine_model_path = "unet_v1.trt"
engine = trt_helper.get_engine(1,'unet_v1.onnx',engine_model_path, fp16_mode=False, int8_mode=False, save_engine=True)
assert engine, 'Broken engine'
context = engine.create_execution_context() 
inputs, outputs, bindings, stream = trt_helper.allocate_buffers(engine)

img = cv2.imread('9.png')
img = torch.from_numpy(img)
inputs = [img]

for i in range(20):
    ta = time.time()
    trt_helper.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    tb = time.time()


    print(f'Took {1/(tb-ta)} fps')

print(len(outputs[0]))

out = trt_helper.postprocess_the_outputs(outputs, (256, 160))
print(out.shape)