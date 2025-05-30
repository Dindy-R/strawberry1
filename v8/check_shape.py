from ultralytics import YOLO

# 加载 PyTorch 模型
model_pt = YOLO("D:/User/Desktop/ultralytics/checkpoint/seg/seg_mAp95_43.6.pt")
model_pt.model.eval()
model_pt.model.to('cuda')

import torch

# 准备输入数据
dummy_input = torch.randn(1, 3, 640, 640).cuda()

# 导出模型为 ONNX 格式
model_onnx_path = "model.onnx"
torch.onnx.export(
    model_pt.model,
    dummy_input,
    model_onnx_path,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
import onnx
from onnxsim import simplify

# 加载 ONNX 模型
model_onnx = onnx.load(model_onnx_path)

# 简化 ONNX 模型
model_simplified, check = simplify(model_onnx)
assert check, "Simplified ONNX model could not be validated"

# 保存简化后的 ONNX 模型
model_simplified_path = "model_simplified.onnx"
onnx.save(model_simplified, model_simplified_path)
import tensorrt as trt

import tensorrt as trt


def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(trt.Logger(trt.Logger.VERBOSE)) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, trt.Logger(trt.Logger.VERBOSE)) as parser:

        # 设置构建器配置
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)

        # 解析 ONNX 模型
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse the ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 构建 TensorRT 引擎
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build the engine.")

        # 保存 TensorRT 引擎
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine


engine_file_path = "D:/User/Desktop/ultralytics/checkpoint/seg/seg_mAp95_43.6.engine"
build_engine(model_simplified_path, engine_file_path)

import numpy as np

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize the engine.")
        return engine

def create_context(engine):
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_tensor_shape(binding)
            break
    context = engine.create_execution_context()
    return context, input_shape

def infer_with_tensorrt(context, input_data):
    output_buffer = np.empty_like(input_data, dtype=np.float32)
    d_input = torch.from_numpy(input_data).cuda()
    d_output = torch.empty_like(d_input).cuda()
    bindings = [d_input.data_ptr(), d_output.data_ptr()]
    context.execute_v2(bindings)
    output_buffer = d_output.cpu().numpy()
    return output_buffer

# 加载 TensorRT 引擎
engine = load_engine(engine_file_path)
context, input_shape = create_context(engine)

# 准备输入数据
dummy_input = torch.randn(input_shape).cuda()

# 运行 PyTorch 模型
with torch.no_grad():
    output_pt = model_pt.predict(dummy_input)
output_pt = output_pt[0].cpu().numpy()  # 假设输出是一个列表，取第一个元素

# 运行 TensorRT 引擎
output_trt = infer_with_tensorrt(context, dummy_input.cpu().numpy())

# 比较输出结果
abs_error = np.abs(output_pt - output_trt)
max_abs_error = np.max(abs_error)
rmse = np.sqrt(np.mean((output_pt - output_trt) ** 2))

print(f"Maximum Absolute Error: {max_abs_error}")
print(f"Root Mean Squared Error: {rmse}")