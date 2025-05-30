from ultralytics import YOLO
import torch
import onnx

# 加载 PyTorch 模型并移动到 GPU
model_pt = YOLO("D:/User/Desktop/ultralytics/checkpoint/seg/seg_mAp95_43.6.pt")
model_pt.model.eval()
model_pt.model.to('cuda')

# 准备输入数据并移动到 GPU
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')

# 运行 PyTorch 模型
with torch.no_grad():
    results = model_pt.predict(dummy_input)

# 检查 PyTorch 模型的输出层结构
# print("PyTorch Model Output Layer Structure:")
# if isinstance(results, (list, tuple)):
#     for i, result in enumerate(results):
#         if hasattr(result, 'boxes'):
#             boxes = result.boxes
#             print(f"Output {i+1} name: boxes")
#             print(f"Output {i+1} shape: {boxes.shape}")
#         if hasattr(result, 'masks'):
#             masks = result.masks
#             print(f"Output {i+1} name: masks")
#             print(f"Output {i+1} shape: {masks.shape}")
#         if hasattr(result, 'keypoints'):
#             keypoints = result.keypoints
#             print(f"Output {i+1} name: keypoints")
#             print(f"Output {i+1} shape: {keypoints.shape}")
# else:
#     if hasattr(results, 'boxes'):
#         boxes = results.boxes
#         print(f"Output name: boxes")
#         print(f"Output shape: {boxes.shape}")
#     if hasattr(results, 'masks'):
#         masks = results.masks
#         print(f"Output name: masks")
#         print(f"Output shape: {masks.shape}")
#     if hasattr(results, 'keypoints'):
#         keypoints = results.keypoints
#         print(f"Output name: keypoints")
#         print(f"Output shape: {keypoints.shape}")

# 导出模型为 ONNX 格式
model_pt.export(format='onnx', imgsz=(640, 640), device='cuda')

# 加载 ONNX 模型
model_onnx_path = "D:/User/Desktop/ultralytics/checkpoint/seg/seg_mAp95_43.6.onnx"
model_onnx = onnx.load(model_onnx_path)

# 检查 ONNX 模型的输出层结构
print("\nONNX Model Output Layer Structure:")
for output in model_onnx.graph.output:
    print(f"Output name: {output.name}")
    print(f"Output type: {output.type.tensor_type}")
    print(f"Output shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
    print()