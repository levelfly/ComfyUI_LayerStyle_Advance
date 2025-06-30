import tensorrt as trt
import os

# --- 設定 ---
# [請修改] 1. 指向您下載的 ONNX 模型檔案的絕對路徑
ONNX_PATH = '/root/ComfyUI/models/BiRefNet/onnx/BiRefNet-general-epoch_244.onnx' # <--- 請務必確認這個路徑是正確的！

# 2. 輸出 TensorRT 引擎的路徑
# 注意：腳本會自動建立 'trt' 資料夾
ENGINE_NAME = 'BiRefNet-general-epoch_244.trt'
ENGINE_PATH = os.path.join('/root/ComfyUI/models/BiRefNet/trt/', ENGINE_NAME)

# 3. 推論與工作空間設定
INFERENCE_SIZE = 1024 # 假設 ONNX 模型是為 1024x1024 設計的
WORKSPACE_GB = 4

print("--- TensorRT Engine Builder (from pre-existing ONNX) ---")

if not os.path.exists(ONNX_PATH):
    print(f"FATAL: Shared ONNX model not found at {ONNX_PATH}")
    exit()

# ======================================================================
# 唯一的階段：從 ONNX 檔案建構 TensorRT 引擎
# ======================================================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)
print(f"FP16 mode enabled: True")

print(f"Parsing ONNX model from {ONNX_PATH}...")
with open(ONNX_PATH, 'rb') as model:
    if not parser.parse(model.read()):
        print("FATAL: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
print("ONNX model parsed successfully.")

profile = builder.create_optimization_profile()
fixed_shape = (1, 3, INFERENCE_SIZE, INFERENCE_SIZE)
# 假設輸入名稱為 'image'，如果報錯，我們再用 inspect_onnx.py 檢查
profile.set_shape("image", min=fixed_shape, opt=fixed_shape, max=fixed_shape)
config.add_optimization_profile(profile)
print(f"Optimization profile set for fixed size {INFERENCE_SIZE}x{INFERENCE_SIZE}.")

print("Building TensorRT engine... This may take several minutes.")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("FATAL: Failed to build the engine.")
    exit()

print("Engine built successfully!")

# --- 已修復的部分 ---
# 在寫入檔案前，先取得目標資料夾的路徑
output_dir = os.path.dirname(ENGINE_PATH)
# 建立目標資料夾，如果它不存在的話 (exist_ok=True 避免在資料夾已存在時報錯)
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_dir}")
# --- 修復結束 ---


with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print(f"\n--- ALL DONE! ---")
print(f"Final TensorRT engine saved to: {ENGINE_PATH}")
