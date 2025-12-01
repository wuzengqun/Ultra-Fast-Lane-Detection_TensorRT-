import torch
import torch.onnx
from model.model import parsingNet

def export_onnx_from_original(weight_path, onnx_path, input_shape=(1, 3, 288, 800)):
    """
    从原始PyTorch模型导出ONNX
    """
    print("从原始PyTorch模型导出ONNX...")
    
    # 1. 加载原始模型
    net = parsingNet(
        pretrained=False,
        backbone='18',
        cls_dim=(200 + 1, 18, 4),  # (griding_num+1, cls_num_per_lane, 4)
        use_aux=False
    )
    
    # 加载权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    # 处理权重键名
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    # 2. 创建虚拟输入
    dummy_input = torch.randn(*input_shape)
    
    # 3. 导出ONNX
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX模型已导出: {onnx_path}")
    return onnx_path

# 使用示例
export_onnx_from_original('culane_18.pth', 'UltraFastLaneDetection.onnx')