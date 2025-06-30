import os
import sys
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import tqdm
from comfy.utils import ProgressBar

# 假設您的環境中已經有 comfy 相關的路徑和函式
# 例如 folder_paths, log 等
# 並且 imagefunc.py 提供了必要的輔助函式
try:
    import folder_paths
    from .imagefunc import *
except ImportError:
    # 為了程式碼在沒有 ComfyUI 環境下也能被靜態分析，提供一些虛設的定義
    class MockFolderPaths:
        def __init__(self):
            self.models_dir = './models'


    folder_paths = MockFolderPaths()


    def log(*args, **kwargs):
        print(*args, **kwargs)


    def tensor2pil(t):
        pass


    def pil2tensor(p):
        pass


    def get_files(p, ext):
        return {}


    def guided_filter_alpha(*args, **kwargs):
        pass


    def mask_edge_detail(*args, **kwargs):
        pass


    def generate_VITMatte_trimap(*args, **kwargs):
        pass


    def generate_VITMatte(*args, **kwargs):
        pass


    def histogram_remap(*args, **kwargs):
        pass

# 確保 BiRefNet_v2 模組路徑正確
sys.path.append(os.path.join(os.path.dirname(__file__), 'BiRefNet_v2'))

# +++ TensorRT 函式庫引入 +++
try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


# +++ 結束區塊 +++


def get_models():
    """從指定路徑獲取 .pth 模型檔案列表"""
    model_path = os.path.join(folder_paths.models_dir, 'BiRefNet', 'pth')
    model_ext = [".pth"]
    model_dict = get_files(model_path, model_ext)
    return model_dict


# +++ (優化) TensorRT 模型執行器輔助類別 +++
# 這個類別會載入 TRT 引擎並封裝推理過程，使其可以像 PyTorch 模型一樣被調用
class TRTWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # 從 .trt 檔案載入引擎
        with open(self.engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.device = torch.device('cuda')

        # 獲取輸入和輸出張量的名稱
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

        print(f"TRT Engine loaded - Inputs: {self.input_names}, Outputs: {self.output_names}")

    def __call__(self, input_tensor):
        # 確保輸入張量在 CUDA 上且是連續的
        input_tensor = input_tensor.to(self.device).contiguous()

        # 設定輸入張量
        input_name = self.input_names[0]  # 假設只有一個輸入
        self.context.set_input_shape(input_name, input_tensor.shape)
        self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        # 準備輸出張量
        output_name = self.output_names[0]  # 假設只有一個輸出
        # 根據輸入的批次大小和模型的輸出規格來決定輸出形狀
        output_shape_from_engine = self.engine.get_tensor_shape(output_name)
        output_shape = (input_tensor.shape[0],) + tuple(output_shape_from_engine)[1:]

        output_tensor = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(output_name, output_tensor.data_ptr())

        # (*** 優化 ***) 使用非同步執行，不在內部呼叫 synchronize
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT inference failed")

        # 返回一個元組，以保持與 PyTorch 模型輸出格式的一致性
        return (output_tensor,)

    def eval(self):
        return self

    def to(self, device):
        return self


class LS_LoadBiRefNetModel:
    def __init__(self):
        self.birefnet = None
        self.state_dict = None

    @classmethod
    def INPUT_TYPES(s):
        tmp_list = list(get_models().keys())
        model_list = []
        if 'BiRefNet-general-epoch_244.pth' in tmp_list:
            model_list.append('BiRefNet-general-epoch_244.pth')
            tmp_list.remove('BiRefNet-general-epoch_244.pth')
        model_list.extend(tmp_list)

        return {
            "required": {
                "model": (model_list,),
            },
        }

    RETURN_TYPES = ("BIREFNET_MODEL",)
    RETURN_NAMES = ("birefnet_model",)
    FUNCTION = "load_birefnet_model"
    CATEGORY = '😺dzNodes/LayerMask'

    def load_birefnet_model(self, model):
        from .BiRefNet_v2.models.birefnet import BiRefNet
        from .BiRefNet_v2.utils import check_state_dict
        model_dict = get_models()
        self.birefnet = BiRefNet(bb_pretrained=False)
        self.state_dict = torch.load(model_dict[model], map_location='cpu', weights_only=True)
        self.state_dict = check_state_dict(self.state_dict)
        self.birefnet.load_state_dict(self.state_dict)
        return (self.birefnet,)


class LS_LoadBiRefNetModelV2:
    def __init__(self):
        self.model = None

    birefnet_model_repos = {
        "BiRefNet-General": "ZhengPeng7/BiRefNet",
        "RMBG-2.0": "briaai/RMBG-2.0"
    }

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(cls.birefnet_model_repos.keys())

        # 自動搜尋所有 .trt 檔案
        trt_folder = os.path.join(folder_paths.models_dir, 'BiRefNet', 'trt')
        if TRT_AVAILABLE and os.path.exists(trt_folder):
            trt_files = glob.glob(os.path.join(trt_folder, '*.trt'))
            for path in trt_files:
                model_name = f"TRT:{os.path.basename(path)}"
                if model_name not in model_list:
                    model_list.append(model_name)

        return {
            "required": {
                "version": (model_list, {"default": model_list[0] if model_list else None}),
            },
        }

    RETURN_TYPES = ("BIREFNET_MODEL",)
    RETURN_NAMES = ("birefnet_model",)
    FUNCTION = "load_birefnet_model"
    CATEGORY = '😺dzNodes/LayerMask'

    def load_birefnet_model(self, version):
        if version == "BiRefNet-TRT (local)":
            trt_model_path = os.path.join(folder_paths.models_dir, 'BiRefNet', 'trt', 'birefnet_from_shared.trt')
            if not TRT_AVAILABLE:
                raise ImportError("TensorRT library is not installed. Please install it to use the TRT model.")
            if not os.path.exists(trt_model_path):
                raise FileNotFoundError(f"TRT model not found at: {trt_model_path}")

            log(f"Loading BiRefNet TensorRT model from {trt_model_path}...")
            self.model = TRTWrapper(engine_path=trt_model_path)
            log("TensorRT model loaded successfully.")
            return (self.model,)

        birefnet_path = os.path.join(folder_paths.models_dir, 'BiRefNet')
        os.makedirs(birefnet_path, exist_ok=True)
        model_path = os.path.join(birefnet_path, version)

        if version == "BiRefNet-General":
            old_birefnet_path = os.path.join(birefnet_path, 'pth')
            old_model = "BiRefNet-general-epoch_244.pth"
            old_model_path = os.path.join(old_birefnet_path, old_model)
            if os.path.exists(old_model_path):
                from .BiRefNet_v2.models.birefnet import BiRefNet
                from .BiRefNet_v2.utils import check_state_dict
                self.birefnet = BiRefNet(bb_pretrained=False)
                self.state_dict = torch.load(old_model_path, map_location='cpu', weights_only=True)
                self.state_dict = check_state_dict(self.state_dict)
                self.birefnet.load_state_dict(self.state_dict)
                return (self.birefnet,)
        elif not os.path.exists(model_path):
            log(f"Downloading {version} model...")
            repo_id = self.birefnet_model_repos[version]
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])

        self.model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        return (self.model,)


class LS_BiRefNetUltraV2:
    def __init__(self):
        self.NODE_NAME = 'BiRefNetUltraV2'

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "birefnet_model": ("BIREFNET_MODEL",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 2, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": False}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "birefnet_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'

    def birefnet_ultra_v2(self, image, birefnet_model, detail_method, detail_erode, detail_dilate,
                          black_point, white_point, process_detail, device, max_megapixels):

        batch_size, orig_h, orig_w, _ = image.shape
        image_bchw = image.permute(0, 3, 1, 2)

        is_trt = isinstance(birefnet_model, TRTWrapper)
        inference_device = 'cuda' if is_trt else device

        if not is_trt:
            torch.set_float32_matmul_precision('high')
            birefnet_model.to(inference_device)
            birefnet_model.eval()

        # (*** 優化 ***) 使用 Tensor 操作進行批次預處理
        inference_image_size = (1024, 1024)
        inference_tensor = torch.nn.functional.interpolate(
            image_bchw.to(inference_device),
            size=inference_image_size,
            mode='bilinear',
            align_corners=False
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=inference_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=inference_device).view(1, 3, 1, 1)
        inference_tensor = (inference_tensor - mean) / std

        # (*** 優化 ***) 整個批次進行一次模型推理
        with torch.no_grad():
            preds_batch = birefnet_model(inference_tensor)[-1]

        # (*** 優化 ***) 使用 Tensor 操作進行批次後處理
        masks_batch = torch.sigmoid(preds_batch).cpu()
        masks_resized_batch = torch.nn.functional.interpolate(
            masks_batch, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        masks_enhanced_batch = torch.clamp(masks_resized_batch * 1.08, 0, 1)
        final_masks_tensor = masks_enhanced_batch

        # (*** 優化 ***) 僅在需要時才對細節處理部分使用迴圈
        if process_detail:
            pbar = ProgressBar(batch_size)
            processed_masks_list = []
            image_pil_list = [tensor2pil(img.unsqueeze(0)) for img in image_bchw.cpu()]

            for i in range(batch_size):
                _mask_tensor = masks_enhanced_batch[i].unsqueeze(0)
                _image_tensor_bchw = image_bchw[i].unsqueeze(0)
                _image_pil = image_pil_list[i]

                if detail_method == 'GuidedFilter':
                    _processed = guided_filter_alpha(_image_tensor_bchw, _mask_tensor, detail_erode + detail_dilate // 6 + 1)
                    _processed = histogram_remap(_processed, black_point, white_point)
                elif detail_method == 'PyMatting':
                    _processed = mask_edge_detail(_image_tensor_bchw, _mask_tensor, detail_erode + detail_dilate // 8 + 1, black_point, white_point)
                elif 'VITMatte' in detail_method:
                    local_files_only = detail_method == 'VITMatte(local)'
                    _trimap_pil = generate_VITMatte_trimap(tensor2pil(_mask_tensor), detail_erode, detail_dilate)
                    _mask_pil = generate_VITMatte(_image_pil, _trimap_pil, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _processed = pil2tensor(_mask_pil)
                    _processed = histogram_remap(_processed, black_point, white_point)
                else:
                    _processed = _mask_tensor

                processed_masks_list.append(_processed)
                pbar.update(1)

            final_masks_tensor = torch.cat(processed_masks_list, dim=0)

        # (*** 優化 ***) 批次化組合輸出
        final_masks_hwc = final_masks_tensor.permute(0, 2, 3, 1)
        ret_images_rgba = torch.cat((image, final_masks_hwc), dim=-1)
        ret_masks = final_masks_tensor.squeeze(1)

        log(f"{self.NODE_NAME} Processed {batch_size} image(s) via batched inference.", message_type='finish')
        return (ret_images_rgba, ret_masks,)


# --- ComfyUI 節點註冊 ---
NODE_CLASS_MAPPINGS = {
    "LayerMask: BiRefNetUltraV2": LS_BiRefNetUltraV2,
    "LayerMask: LoadBiRefNetModel": LS_LoadBiRefNetModel,
    "LayerMask: LoadBiRefNetModelV2": LS_LoadBiRefNetModelV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: BiRefNetUltraV2": "LayerMask: BiRefNet Ultra V2",
    "LayerMask: LoadBiRefNetModel": "LayerMask: Load BiRefNet Model",
    "LayerMask: LoadBiRefNetModelV2": "LayerMask: Load BiRefNet Model V2"
}