# Ultra-Fast-Lane-Detection_TensorRT-加速
---
本仓库存储了Ultra-Fast-Lane-Detection算法TensorRT加速的示例代码。
由于模型较大，无法放在仓库中，完整代码自取：[完整代码](https://github.com/wuzengqun/Ultra-Fast-Lane-Detection_TensorRT-/releases/download/v1.0.0/Ultra-Fast-Lane-Detection_tensorrt_local.zip)  


演示视频:  
---
<video src="https://github.com/user-attachments/assets/e38b803f-186a-4bdc-b90e-60e7625284cf" controls width="400">
  Your browser does not support the video tag.
</video>  


文件说明:
---
pt2onnx.py：将pt转onnx  

onnx2trt.py：将onnx转tensorrt（tensorrt8.6.1）  

tensorRT_inferenc_demo.py：tensorRT模型图片推理demo  

tensorRT_inferenc_video.py：tensorRT模型视频推理demo（以图片文件夹形式，下载地址: https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu ，选择video_example.zip）  


导出onnx:  
---
1、下载Ultra-Fast-Lane-Detection官方源码：https://github.com/cfzd/Ultra-Fast-Lane-Detection  
2、下载CULane数据集训练得到的Ultra-Fast-Lane-Detection模型，官方地址有提供  
3、将pt2onnx.py放到Ultra-Fast-Lane-Detection文件夹中：  
```bash
python onnx2trt.py
```
4、将转换得到的onnx文件放到本仓库代码文件夹中即可  

Acknowledgements  
---
This project is based on  
 ● https://github.com/cqu20160901/Ultra-Fast-Lane-Detection_caffe_onnx_horizon_rknn  
 ● https://github.com/cfzd/Ultra-Fast-Lane-Detection  
Thanks to the original authors for their work!  








