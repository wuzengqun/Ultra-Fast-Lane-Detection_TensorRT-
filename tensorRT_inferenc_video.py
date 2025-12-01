import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from scipy.interpolate import CubicSpline

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

input_h = 288
input_w = 800
output_w = 200

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def get_engine_from_bin(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(img_src):
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = img / 255.
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

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


def do_inference(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    stream.synchronize()
    return outputs[0].host

def main():
    engine_file_path = './UltraFastLaneDetection_fp32.trt'
    img_folder = './test_pic/' 
    output_video_path = './lane_output.mp4'  

    img_list = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(img_list) == 0:
        print("没有找到图片")
        return

    sample_frame = cv2.imread(img_list[0])
    height, width, _ = sample_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4编码
    out = cv2.VideoWriter(output_video_path, fourcc, 60.0, (width, height))  

    engine = get_engine_from_bin(engine_file_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    sample_w = (input_w - 1) / (output_w - 1)

    for img_path in img_list:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        img_h, img_w = frame.shape[:2]
        image = preprocess(frame)
        np.copyto(inputs[0].host, image.ravel())

        trt_output = do_inference(context, bindings, inputs, outputs, stream)
        trt_out = trt_output.reshape(1, 201, 18, 4)
        result = postprocess(trt_out[0])

        # 绘制车道线
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        for i in range(result.shape[1]):  # 4 条车道线
            lane_points = []

            for k in range(result.shape[0]):
                if result[k, i] > 0:
                    x = int(result[k, i] * sample_w * img_w / input_w)
                    y = int(img_h - k * 20)
                    lane_points.append((x, y))

                    # （可选）原始点
                    # cv2.circle(frame, (x, y), 5, colors[i], -1)

            # 至少需要 4 个点才能拟合三次样条
            if len(lane_points) < 4:
                continue
            lane_points.sort(key=lambda p: p[1])

            xs = np.array([p[0] for p in lane_points])
            ys = np.array([p[1] for p in lane_points])

            spline = CubicSpline(ys, xs)
            y_min, y_max = ys.min(), ys.max()
            y_dense = np.linspace(y_min, y_max, 120)  # 120个点，非常平滑
            x_dense = spline(y_dense)

            curve_points = np.array([(int(x), int(y)) for x, y in zip(x_dense, y_dense)], dtype=np.int32)

            cv2.polylines(frame, [curve_points], isClosed=False, color=colors[i], thickness=3)


        # 写入视频
        # out.write(frame)

        cv2.imshow('Lane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"视频已保存到 {output_video_path}")


if __name__ == '__main__':
    main()
