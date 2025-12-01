import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

color_list = [(100, 149, 237), (0, 0, 255), (173, 255, 47), (240, 255, 255), (0, 100, 0),
              (47, 79, 79), (255, 228, 196), (138, 43, 226), (165, 42, 42), (222, 184, 135)]

lane_Id_type = [7, 5, 3, 1, 2, 4, 6, 8]

input_h = 288
input_w = 800

output_w = 200

line_type = ['No lane markings',
             'Single white solid line',
             'Single white dashed line',
             'Single solid yellow line',
             'Single yellow dashed line',
             'Double solid white lines',
             'Double solid yellow lines',
             'Double yellow dashed lines',
             'Double white yellow solid lines',
             'Double white dashed lines',
             'Double white solid dashed lines']

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def preprocess(img_src):
    # 与 ONNXRuntime 一致的预处理
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 288))
    img = img / 255.
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加 batch 维度
    return img


def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    value = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return value

def postprocess(output):
    result = np.zeros(shape=(18, 4))
    for i in range(4):
        for j in range(18):
            total = 0
            maxvalue = 0
            maxindex = 0
            for k in range(200):
                if maxvalue < output[k, j, i]:
                    maxvalue = output[k, j, i]
                    maxindex = k
                if k == 199:
                    if maxvalue < output[k + 1, j, i]:
                        maxvalue = output[k + 1, j, i]
                        maxindex = k

                tmp = exp(output[k, j, i])
                total += tmp

            for k in range(200):
                if maxindex < 199:
                    tmp = exp(output[k, j, i]) / total
                    output[k, j, i] = tmp
                    result[17 - j, i] += tmp * (k + 1)

    return result

def main():
    engine_file_path = './UltraFastLaneDetection_fp32.trt'
    input_image_path = 'test.jpg'

    src_img = cv2.imread(input_image_path)
    img_h, img_w = src_img.shape[:2]
    print('img_h, img_w',img_h, img_w)
    
    # TensorRT 输入
    image = preprocess(src_img)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        
        # 将预处理好的图片拷贝到输入 buffer
        np.copyto(inputs[0].host, image.ravel())

        import time
        start_time = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        trt_out = trt_outputs[0].reshape(1, 201, 18, 4)  # [batch, 201, 18, 4]对应griding_num+1、cls_num_per_lane、lane_num
        end_time = time.time()
        print("Inference time: {:.3f} ms".format((end_time - start_time)*1000))

        # trt_outputs[0] 是 [1,201,18,4]，去掉 batch 维度再 postprocess
        result = postprocess(trt_out[0])

        # 绘制车道点
        sample_w = (input_w - 1) / (output_w - 1)   # sample_w = 799/199    output_w就是对应griding_num
        print('output_w', output_w)
        print('result.shape[1] ', result.shape[1])  # result.shape[1] = 4,也就是lane_num
        print('result.shape[0] ', result.shape[0])  # result.shape[0] = 18,也就是cls_num_per_lane
        for i in range(result.shape[1]):
            for k in range(result.shape[0]):
                if result[k, i] > 0:
                    point = (int(result[k, i] * sample_w * img_w / input_w) - 1, int(img_h - k * 20) - 1)
                    if i == 0:
                        cv2.circle(src_img, point, 5, (255, 0, 0), -1)
                    if i == 1:
                        cv2.circle(src_img, point, 5, (0, 255, 0), -1)
                    if i == 2:
                        cv2.circle(src_img, point, 5, (0, 0, 255), -1)
                    if i == 3:
                        cv2.circle(src_img, point, 5, (0, 255, 255), -1)

        cv2.imwrite('./test_result_onnx.jpg', src_img)


if __name__ == '__main__':
    print('This is main ...')
    main()
