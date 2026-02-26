# ComfyUI-QwenVL (aistudynow edition)
# Custom nodes for Qwen-VL / Qwen3-VL / Qwen2.5-VL inside ComfyUI
# Author: aistudynow
# License: GPL-3.0

import gc
import json
import os
import platform
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import psutil  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from huggingface_hub import snapshot_download, HfApi, hf_hub_download  # type: ignore
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq  # type: ignore
except ImportError:
    from transformers import AutoModelForVision2Seq  # type: ignore
from transformers import AutoProcessor  # type: ignore
from transformers import AutoTokenizer, BitsAndBytesConfig  # type: ignore
import folder_paths  # type: ignore  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    from comfy.utils import ProgressBar  # type: ignore
except Exception:
    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, value, total=None, preview=None):
            _ = (value, total, preview)

try:
    from sageattention.core import (  # type: ignore
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except Exception:
    SAGE_ATTENTION_AVAILABLE = False

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"


def load_model_configs():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
    except json.JSONDecodeError:
        print("Error: Failed to parse configuration file.")
    return {}


MODEL_CONFIGS = load_model_configs()


class Quantization(str, Enum):
    Q4_BIT = "4-bit (VRAM-friendly)"
    Q8_BIT = "8-bit (Balanced)"
    NONE = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]


ATTENTION_MODES = ["auto", "sage", "flash_attention_2", "sdpa"]


def get_model_info(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name, {})


def get_device_info() -> dict:
    gpu_info = {"available": False, "total_memory": 0.0, "free_memory": 0.0}
    device_type = "cpu"
    recommended_device = "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {
            "available": True,
            "total_memory": total_mem,
            "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3),
        }
        device_type = "nvidia_gpu"
        recommended_device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended_device = "mps"

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {
        "total": sys_mem.total / 1024**3,
        "available": sys_mem.available / 1024**3,
    }

    memory_sufficient = True
    warning_message = ""
    if recommended_device == "mps" and sys_mem_info["total"] < 16:
        memory_sufficient = False
        warning_message = "Apple Silicon memory is less than 16GB, performance may be affected."
    elif recommended_device == "cuda" and gpu_info["total_memory"] < 8:
        memory_sufficient = False
        warning_message = "GPU VRAM is less than 8GB, performance may be degraded."

    return {
        "gpu": gpu_info,
        "system_memory": sys_mem_info,
        "device_type": device_type,
        "recommended_device": recommended_device,
        "memory_sufficient": memory_sufficient,
        "warning_message": warning_message,
    }


def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"

    if device.isdigit():
        device = f"cuda:{int(device)}"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but unavailable, falling back to CPU")
            return "cpu"
        return "cuda"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but unavailable, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"[QwenVL] CUDA device {device_idx} unavailable, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                print(f"[QwenVL] Invalid CUDA device format '{device}', using cuda:0")
                return "cuda:0"
        return device

    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL] MPS requested but unavailable, falling back to CPU")
            return "cpu"
        return "mps"

    return "cpu" if device not in {"cpu"} else device


def get_device_options() -> list[str]:
    options = ["auto", "cuda", "cpu", "mps"]
    if torch.cuda.is_available():
        options.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return list(dict.fromkeys(options))


def check_memory_requirements(
    model_name: str,
    quantization: str,
    device_info: dict,
    requested_device: str | None = None,
) -> str:
    model_info = get_model_info(model_name)
    vram_req = model_info.get("vram_requirement", {})
    quant_map = {
        Quantization.Q4_BIT.value: vram_req.get("4bit", 0.0),
        Quantization.Q8_BIT.value: vram_req.get("8bit", 0.0),
        Quantization.NONE.value: vram_req.get("full", 0.0),
    }

    base_memory = quant_map.get(quantization, 0.0)
    device = requested_device or device_info.get("recommended_device", "cpu")
    use_cpu_mps = device in ["cpu", "mps"]

    required_mem = base_memory * (1.5 if use_cpu_mps else 1.0)
    
    available_mem = device_info.get("system_memory", {}).get("available", 0.0) if use_cpu_mps else device_info.get("gpu", {}).get("free_memory", 0.0)
    if not isinstance(available_mem, (float, int)):
        available_mem = 0.0
    mem_type = "System RAM" if use_cpu_mps else "GPU VRAM"

    if required_mem * 1.2 > available_mem:
        print(f"Warning: Insufficient {mem_type} ({available_mem:.2f}GB available). Lowering quantization...")
        if quantization == Quantization.NONE.value:
            return Quantization.Q8_BIT.value
        if quantization == Quantization.Q8_BIT.value:
            return Quantization.Q4_BIT.value
        raise RuntimeError(f"Insufficient {mem_type} even for 4-bit quantization.")
    return quantization


def is_fp8_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "-fp8" in lowered or "_fp8" in lowered


def flash_attn_available() -> bool:
    if platform.system() != "Linux":
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    try:
        import flash_attn  # type: ignore # noqa: F401
    except Exception:
        return False
    try:
        import importlib.metadata as importlib_metadata
        _ = importlib_metadata.version("flash_attn")
    except Exception:
        return False
    return True


def sage_attn_available() -> bool:
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_sage_attention_config():
    if not sage_attn_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor

    if arch_code >= 120:
        print("[QwenVL] SageAttention: Using SM120 (Blackwell) FP8 kernel")
        return sageattn_qk_int8_pv_fp8_cuda, "per_warp", "fp32+fp32"
    if arch_code >= 90:
        print("[QwenVL] SageAttention: Using SM90 (Hopper) FP8 kernel")
        return sageattn_qk_int8_pv_fp8_cuda_sm90, "per_warp", "fp32+fp32"
    if arch_code == 89:
        print("[QwenVL] SageAttention: Using SM89 (Ada) FP8 kernel")
        return sageattn_qk_int8_pv_fp8_cuda, "per_warp", "fp32+fp32"
    if arch_code >= 80:
        print("[QwenVL] SageAttention: Using SM80+ (Ampere) FP16 kernel")
        return sageattn_qk_int8_pv_fp16_cuda, "per_warp", "fp32"
    print(f"[QwenVL] SageAttention not supported on SM{arch_code}")
    return None, None, None


def resolve_attention_mode(mode: str, force_sdpa: bool = False) -> str:
    # Handle legacy boolean values if they slip through
    if str(mode) == "True":
        mode = "auto"
    if str(mode) == "False":
        mode = "sdpa"

    if force_sdpa:
        return "sdpa"

    if mode == "sdpa":
        return "sdpa"
    if mode == "sage":
        if sage_attn_available():
            return "sage"
        print("[QwenVL] SageAttention forced but unavailable, falling back to SDPA")
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    if sage_attn_available():
        print("[QwenVL] Auto mode: Using SageAttention")
        return "sage"
    if flash_attn_available():
        print("[QwenVL] Auto mode: Using Flash Attention 2")
        return "flash_attention_2"
    print("[QwenVL] Auto mode: Using SDPA")
    return "sdpa"


def set_sage_attention(model):
    if not sage_attn_available():
        raise ImportError("SageAttention is not installed or this GPU is unsupported.")

    sage_attn_func, qk_quant_gran, pv_accum_dtype = get_sage_attention_config()
    if sage_attn_func is None:
        raise RuntimeError("No compatible SageAttention kernel found for this GPU.")

    attention_classes = []
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention,  # type: ignore
            apply_rotary_pos_emb as qwen2_apply_rotary,
        )
        attention_classes.append((Qwen2Attention, qwen2_apply_rotary))
    except Exception:
        pass

    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention,  # type: ignore
            apply_rotary_pos_emb as qwen3_apply_rotary,
        )
        attention_classes.append((Qwen3Attention, qwen3_apply_rotary))
    except Exception:
        pass

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextAttention,  # type: ignore
            apply_rotary_pos_emb as qwen3vl_apply_rotary,
        )
        attention_classes.append((Qwen3VLTextAttention, qwen3vl_apply_rotary))
    except Exception:
        pass

    if not attention_classes:
        print("[QwenVL] Could not import compatible Qwen attention classes for SageAttention patching")
        return

    def make_sage_forward(attention_class, apply_rotary_pos_emb_func):
        def sage_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple | None = None,
            attention_mask: torch.Tensor | None = None,
            past_key_values=None,
            cache_position: torch.LongTensor | None = None,
            position_ids: torch.LongTensor | None = None,
            **kwargs,
        ):
            _ = position_ids
            original_dtype = hidden_states.dtype

            is_4bit = hasattr(self.q_proj, "quant_state")
            target_dtype = torch.bfloat16 if is_4bit else self.q_proj.weight.dtype

            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q_len = input_shape[1] if len(input_shape) > 1 else hidden_states.size(1)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            else:
                query_states = query_states.view(hidden_shape).transpose(1, 2)

            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            else:
                key_states = key_states.view(hidden_shape).transpose(1, 2)

            value_states = value_states.view(hidden_shape).transpose(1, 2)

            sin = None
            cos = None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_func(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin if position_embeddings else None,
                    "cos": cos if position_embeddings else None,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

            is_causal = attention_mask is None and q_len > 1
            attn_output = sage_attn_func(  # type: ignore
                query_states.to(target_dtype),
                key_states.to(target_dtype),
                value_states.to(target_dtype),
                tensor_layout="HND",
                is_causal=is_causal,
                qk_quant_gran=qk_quant_gran,
                pv_accum_dtype=pv_accum_dtype,
            )

            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)

            if attn_output.dtype != original_dtype:
                attn_output = attn_output.to(original_dtype)

            return attn_output, None

        return sage_attention_forward

    patched_count = 0
    for attention_class, apply_rotary_func in attention_classes:
        sage_forward = make_sage_forward(attention_class, apply_rotary_func)
        for module in model.modules():
            # Bypass Pyre "Expected class object" by checking type name or using type.__name__
            if type(module).__name__ == attention_class.__name__:
                setattr(module, "__class__", attention_class)  # type: ignore
                setattr(module, "forward", sage_forward.__get__(module, attention_class))  # type: ignore
                patched_count += 1

    if patched_count > 0:
        print(f"[QwenVL] SageAttention: patched {patched_count} attention layers")
    else:
        print("[QwenVL] SageAttention: no compatible attention layers found")


def get_model_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for dev in hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev not in {"disk", "meta"}:
                return torch.device(dev)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def _human_bytes(n_bytes: int | None = None) -> str:
    if n_bytes is None:
        return ""
    if n_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n_bytes)
    for u in units:
        if s < 1024 or u == "TB":
            return f"{s:.2f} {u}"
        s /= 1024.0
    return ""


def _has_required_weights(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    single = model_path / "model.safetensors"
    if single.exists():
        return True
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        for p in model_path.iterdir():
            name = p.name
            if name.startswith("model-") and name.endswith(".safetensors"):
                return True
    for p in model_path.iterdir():
        name = p.name
        if name.startswith("model-") and name.endswith(".safetensors"):
            return True
    return False


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def discover_custom_qwen_text_encoders() -> list[str]:
    try:
        text_encoder_files = folder_paths.get_filename_list("text_encoders")
    except Exception:
        return []

    candidates: list[str] = []
    for file in text_encoder_files:
        if file.lower().endswith(".safetensors") and "qwen" in file.lower():
            candidates.append(f"text_encoders/{Path(file).stem}")
    return sorted(set(candidates))


def infer_base_model_name(custom_model_name: str) -> str | None:
    raw_name = custom_model_name.replace("text_encoders/", "")
    normalized = _normalize_name(raw_name)

    for base_name, info in MODEL_CONFIGS.items():
        if base_name.startswith("_"):
            continue
        base_normalized = _normalize_name(base_name)
        repo_normalized = _normalize_name(info.get("repo_id", ""))
        if base_normalized in normalized or repo_normalized in normalized:
            return base_name

    if "qwen2.5-vl" in normalized:
        candidate = "Qwen2.5-VL-7B-Instruct" if "7b" in normalized else "Qwen2.5-VL-3B-Instruct"
        return candidate if candidate in MODEL_CONFIGS else None

    if "qwen3-vl" in normalized:
        if "8b" in normalized:
            candidate = "Qwen3-VL-8B-Instruct"
        else:
            default_val = MODEL_CONFIGS.get("_default_model", "Qwen3-VL-4B-Instruct")
            candidate = str(default_val) if isinstance(default_val, str) else "Qwen3-VL-4B-Instruct"
        return candidate if candidate in MODEL_CONFIGS else None

    return None


def resolve_custom_weight_path(custom_model_name: str) -> str:
    model_name = custom_model_name.replace("text_encoders/", "")
    try:
        full_path = folder_paths.get_full_path("text_encoders", f"{model_name}.safetensors")
    except Exception:
        return ""

    if full_path and os.path.exists(full_path):
        return full_path
    return ""


class ImageProcessor:
    def to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)


class ModelDownloader:
    def __init__(self1, configs):
        self1.configs = configs
        self1.models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        self1.models_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _infer_model_type(repo_id: str) -> str | None:
        repo_lower = repo_id.lower()

        if "qwen3-vl" in repo_lower:
            return "qwen3_vl"
        if "qwen2.5-vl" in repo_lower:
            return "qwen2_5_vl"
        return None

    def _ensure_model_type(self1, model_path: Path, repo_id: str):
        config_file = model_path / "config.json"
        if not config_file.exists():
            print(f"[aistudynow] Warning: config.json missing for {repo_id} in {model_path}")
            return

        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[aistudynow] Warning: failed to read config.json for {repo_id}: {e}")
            return

        if config_data.get("model_type"):
            return

        inferred_type = self1._infer_model_type(repo_id)
        if inferred_type is None:
            print(
                f"[aistudynow] Warning: Unable to infer model_type for {repo_id}. Please update config.json manually."
            )
            return

        config_data["model_type"] = inferred_type
        try:
            config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
            print(f"[aistudynow] Added missing model_type='{inferred_type}' to {config_file}")
        except Exception as e:
            print(f"[aistudynow] Warning: failed to update config.json for {repo_id}: {e}")

    def _print_transfer_hint(self1):
        if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").strip():
            print(
                "[aistudynow] Tip: enable faster downloads by installing hf_transfer and setting "
                "HF_HUB_ENABLE_HF_TRANSFER=1"
            )

    def ensure_model_available(self1, model_name: str, repo_id_override: str | None = None) -> str:
        model_info = self1.configs.get(model_name)
        repo_id = repo_id_override or (model_info["repo_id"] if model_info else None)
        if not repo_id:
            raise ValueError(f"Model '{model_name}' not found in configuration and no repo_id override provided.")

        model_folder_name = repo_id.split("/")[-1]
        model_path = self1.models_dir / model_folder_name
        model_path.mkdir(parents=True, exist_ok=True)

        if _has_required_weights(model_path):
            self1._ensure_model_type(model_path, repo_id)
            print(f"[aistudynow] Model '{model_name}' ready at {model_path}.")
            return str(model_path)

        self1._print_transfer_hint()

        siblings, total_size = [], None
        try:
            api = HfApi()
            info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = info.siblings or []
            total_size = sum([(s.size or 0) for s in siblings])
            print(f"[aistudynow] {repo_id}: about {_human_bytes(total_size)} across {len(siblings)} files.")
        except Exception as e:
            print(f"[aistudynow] Could not fetch file list/size: {e}")

        def _wanted(s):
            fn = s.rfilename
            if fn == "model.safetensors" or fn == "model.safetensors.index.json":
                return True
            if fn.startswith("model-") and fn.endswith(".safetensors"):
                return True
            if fn in {
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "chat_template.json",
            }:
                return True
            return False

        if siblings:
            files_to_get = [s for s in siblings if _wanted(s)]
            if not files_to_get:
                files_to_get = siblings

            print(f"[aistudynow] Ensuring {len(files_to_get)} files exist in {model_path} ...")

            total_known = sum([(s.size or 0) for s in files_to_get]) if files_to_get else None
            use_bar = bool(tqdm) and isinstance(total_known, int) and total_known > 0
            bar = tqdm(total=total_known, unit="B", unit_scale=True, desc="[aistudynow] Total", leave=False) if use_bar else None  # type: ignore

            for i, s in enumerate(files_to_get, 1):
                local_file = model_path / s.rfilename
                size = getattr(s, "size", None)
                size_txt = _human_bytes(size)

                if local_file.exists() and (size is None or local_file.stat().st_size == size):
                    if bar and size:
                        bar.update(size)  # type: ignore
                    print(f"[aistudynow] [{i}/{len(files_to_get)}] {s.rfilename} (exists, {size_txt})")
                    continue

                print(f"[aistudynow] [{i}/{len(files_to_get)}] {s.rfilename}  ({size_txt})")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=s.rfilename,
                        local_dir=str(model_path),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    if bar and size:
                        bar.update(size)  # type: ignore
                except Exception as e:
                    print(f"[aistudynow]   failed: {e}")

            if bar:
                bar.close()  # type: ignore
        else:
            print(f"[aistudynow] Fallback to snapshot_download for {repo_id} ...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", ".git*"],
                resume_download=True,
            )
            print("[aistudynow] Snapshot download finished.")

        if not _has_required_weights(model_path):
            raise RuntimeError(
                f"Model files incomplete at {model_path}. Missing 'model.safetensors' or the index+shards. "
                f"Try deleting the folder and rerunning so it redownloads clean."
            )

        self1._ensure_model_type(model_path, repo_id)

        print(f"[aistudynow] Model '{model_name}' ready at {model_path}.")
        return str(model_path)


class aistudynow_QwenVL_Advanced:
    CATEGORY = "🧠aistudynow/QwenVL"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "process"

    def __init__(self):
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.current_model_name: str | None = None
        self.current_quantization: str | None = None
        self.current_device: str | None = None
        self.current_weights_path: str | None = None
        self.current_attention_mode: str | None = None
        self.current_use_torch_compile: bool = False
        self.current_signature: tuple | None = None
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()

        print(f"QwenVL Node Initialized. Device: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"Warning: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        if self.model is not None:
            print("[aistudynow] Releasing model resources...")
            try:
                self.model = self.model.cpu()
            except Exception:
                pass
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.current_weights_path = None
        self.current_attention_mode = None
        self.current_use_torch_compile = False
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def _load_fp8_weights_if_needed(model, model_path: str):
        try:
            has_meta = any(param.device.type == "meta" for param in model.parameters())
        except Exception:
            has_meta = False
        if not has_meta:
            return model

        print("[aistudynow] FP8 model has meta tensors, materializing weights...")
        model = model.to_empty(device="cpu")
        try:
            from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict  # type: ignore
            from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME  # type: ignore

            index_file = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                print("[aistudynow] Detected sharded checkpoint, loading all shards...")
                load_sharded_checkpoint(model, model_path, strict=True)
                return model

            if os.path.exists(os.path.join(model_path, SAFE_WEIGHTS_NAME)):
                state_dict_path = os.path.join(model_path, SAFE_WEIGHTS_NAME)
            elif os.path.exists(os.path.join(model_path, WEIGHTS_NAME)):
                state_dict_path = os.path.join(model_path, WEIGHTS_NAME)
            else:
                raise RuntimeError(f"Could not find model weights in {model_path}")

            print(f"[aistudynow] Loading FP8 weights from {state_dict_path}")
            state_dict = load_state_dict(state_dict_path)
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                print(f"[aistudynow] Strict FP8 loading failed ({exc}), retrying non-strict.")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"[aistudynow] Warning: missing {len(missing_keys)} keys in FP8 load.")
                if unexpected_keys:
                    print(f"[aistudynow] Warning: unexpected {len(unexpected_keys)} keys in FP8 load.")
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to materialize FP8 meta tensors: {exc}") from exc

    def load_model(
        self,
        model_name: str,
        quantization_str: str,
        device: str = "auto",
        attention_mode: str = "auto",
        use_torch_compile: bool = False,
    ):
        device_choice = normalize_device_choice(device)
        effective_device = self.device_info["recommended_device"] if device_choice == "auto" else device_choice

        custom_weights_path = None
        base_model_name = model_name
        if model_name.startswith("text_encoders/"):
            base_model_name = infer_base_model_name(model_name)
            if not base_model_name:
                raise ValueError(
                    f"Unsupported custom model '{model_name}'. "
                    "File name must include a known QwenVL base model such as Qwen2.5-VL-3B or Qwen2.5-VL-7B."
                )
            custom_weights_path = resolve_custom_weight_path(model_name)
            if not custom_weights_path:
                raise FileNotFoundError(
                    f"Custom weights for '{model_name}' were not found in the text_encoders folder. "
                    "Ensure the .safetensors file exists and matches the selected name."
                )
            print(f"[aistudynow] Resolved custom weights '{model_name}' -> base model '{base_model_name}'.")

        base_model_info = get_model_info(base_model_name)
        if not base_model_info:
            raise ValueError(f"Model '{base_model_name}' not found in configuration.")

        is_prequantized_fp8 = base_model_info.get("quantized", False) or is_fp8_model_name(base_model_name)
        adjusted_quantization = quantization_str
        if not is_prequantized_fp8:
            adjusted_quantization = check_memory_requirements(
                base_model_name,
                quantization_str,
                self.device_info,
                requested_device=effective_device,
            )

        if not str(effective_device).startswith("cuda") and adjusted_quantization in (
            Quantization.Q4_BIT,
            Quantization.Q8_BIT,
        ):
            print("[aistudynow] 4-bit/8-bit quantization needs CUDA. Falling back to FP16.")
            adjusted_quantization = Quantization.NONE

        is_bnb_quantization = (not is_prequantized_fp8) and adjusted_quantization in (
            Quantization.Q4_BIT,
            Quantization.Q8_BIT,
        )
        force_sdpa = is_prequantized_fp8 or is_bnb_quantization
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)

        if force_sdpa and attention_mode in ["auto", "sage", "flash_attention_2"]:
            if is_prequantized_fp8:
                print("[QwenVL] FP8 model detected - forcing SDPA attention")
            elif is_bnb_quantization:
                print("[QwenVL] BitsAndBytes quantization detected - forcing SDPA attention")

        signature = (
            model_name,
            adjusted_quantization,
            effective_device,
            custom_weights_path or "",
            attn_impl,
            bool(use_torch_compile),
        )
        if self.model is not None and self.current_signature == signature:
            return

        self.clear_model_resources()

        model_path = self.downloader.ensure_model_available(
            base_model_name, repo_id_override=base_model_info.get("repo_id")
        )

        quant_config = None
        load_dtype = None if is_prequantized_fp8 else torch.float16
        if not is_prequantized_fp8:
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_dtype = None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_dtype = None
            elif effective_device == "cpu":
                load_dtype = torch.float32

        actual_attn_impl = "sdpa" if attn_impl == "sage" else attn_impl

        if is_prequantized_fp8:
            if device_choice == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda:0"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                target_device = effective_device
            if target_device == "cuda":
                target_device = "cuda:0"

            load_kwargs_fp8: dict[str, Any] = {
                "attn_implementation": "sdpa",
                "device_map": None,
                "torch_dtype": "auto",
                "use_safetensors": True,
            }
            print(f"[aistudynow] Loading FP8 model '{model_name}' to {target_device}...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                **load_kwargs_fp8,
            )
            self.model = self._load_fp8_weights_if_needed(self.model, model_path)
            self.model = self.model.to(target_device).eval()
            print(f"[aistudynow] FP8 model loaded on {target_device}")
        else:
            load_kwargs: dict[str, Any] = {
                "attn_implementation": actual_attn_impl,
                "use_safetensors": True,
                "trust_remote_code": True,
            }
            if load_dtype is not None:
                load_kwargs["torch_dtype"] = load_dtype
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config

            if device_choice == "auto":
                load_kwargs["device_map"] = "auto"
            elif effective_device in {"cpu", "mps"}:
                load_kwargs["device_map"] = None
            elif effective_device == "cuda":
                load_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif str(effective_device).startswith("cuda:"):
                try:
                    load_kwargs["device_map"] = {"": int(str(effective_device).split(":", 1)[1])}
                except (ValueError, IndexError):
                    load_kwargs["device_map"] = {"": 0}
            else:
                load_kwargs["device_map"] = effective_device

            print(
                f"[aistudynow] Loading model '{model_name}' "
                f"(quant={adjusted_quantization}, attn={attn_impl}, device={effective_device})..."
            )
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
            if load_kwargs["device_map"] is None and effective_device in {"cpu", "mps"}:
                self.model = self.model.to(effective_device)
            self.model = self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if custom_weights_path:
            print(f"[aistudynow] Applying custom weights from {custom_weights_path}")
            try:
                from safetensors.torch import load_file  # type: ignore
            except Exception as exc:
                raise RuntimeError("Failed to import safetensors for loading custom weights.") from exc

            custom_state = load_file(custom_weights_path)
            missing_keys, unexpected_keys = self.model.load_state_dict(custom_state, strict=False)
            if missing_keys:
                print(f"[aistudynow] Warning: missing {len(missing_keys)} keys when loading custom weights.")
            if unexpected_keys:
                print(f"[aistudynow] Warning: {len(unexpected_keys)} unexpected keys in custom weights.")

        if attn_impl == "sage":
            try:
                set_sage_attention(self.model)
                print("[QwenVL] SageAttention enabled")
            except Exception as exc:
                print(f"[QwenVL] SageAttention patching failed: {exc}")

        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True

        if bool(use_torch_compile) and str(effective_device).startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")

        self.current_model_name = model_name
        self.current_quantization = adjusted_quantization
        self.current_device = effective_device
        self.current_weights_path = custom_weights_path
        self.current_attention_mode = attn_impl
        self.current_use_torch_compile = bool(use_torch_compile)
        self.current_signature = signature
        print("[aistudynow] Model loaded successfully.")

    @classmethod
    def INPUT_TYPES(cls):
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_")]
        default_model = next(
            (name for name in model_names if MODEL_CONFIGS[name].get("default")), model_names[0] if model_names else ""
        )
        custom_models = discover_custom_qwen_text_encoders()
        available_models = model_names + [m for m in custom_models if m not in model_names]
        if not default_model and available_models:
            default_model = available_models[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])
        device_options = get_device_options()

        return {
            "required": {
                "model_name": (available_models, {"default": default_model}),
                "quantization": (list(Quantization.get_values()), {"default": Quantization.Q8_BIT}),
                "preset_prompt": (preset_prompts, {"default": preset_prompts[0]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Custom prompt"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "device": (device_options, {"default": "auto"}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto"}),
            },
            "optional": {"image": ("IMAGE",), "video": ("IMAGE",)},
        }

    @torch.no_grad()
    def process(
        self,
        model_name,
        quantization,
        preset_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        num_beams,
        frame_count,
        device,
        use_torch_compile,
        keep_model_loaded,
        seed,
        attention_mode,
        custom_prompt="",
        image=None,
        video=None,
    ):
        start_time = time.time()
        pbar = ProgressBar(3)
        try:
            print("[aistudynow] process(): start")
            torch.manual_seed(seed)
            pbar.update_absolute(1, 3, None)

            print(
                "[aistudynow] process(): load_model("
                f"model={model_name}, quant={quantization}, device={device}, attention={attention_mode}, "
                f"compile={use_torch_compile})"
            )
            self.load_model(
                model_name=model_name,
                quantization_str=quantization,
                device=device,
                attention_mode=attention_mode,
                use_torch_compile=use_torch_compile,
            )
            pbar.update_absolute(2, 3, None)

            final_prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else preset_prompt
            print(f"[aistudynow] process(): final_prompt='{final_prompt[:80]}'")

            conversation: list[dict[str, Any]] = [{"role": "user", "content": []}]

            if image is not None:
                print("[aistudynow] process(): got image input")
                conversation[0]["content"].append({"type": "image", "image": self.image_processor.to_pil(image)})

            if video is not None:
                print("[aistudynow] process(): got video input")
                video_frames = [Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8)) for frame in video]
                if len(video_frames) > frame_count:
                    indices = np.linspace(0, len(video_frames) - 1, frame_count, dtype=int)
                    video_frames = [video_frames[i] for i in indices]
                if len(video_frames) == 1:
                    video_frames.append(video_frames[0])
                conversation[0]["content"].append({"type": "video", "video": video_frames})

            conversation[0]["content"].append({"type": "text", "text": final_prompt})

            print("[aistudynow] process(): building processor inputs")
            text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            pil_images = []
            video_frames_list = []
            for item in conversation[0]["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image" and "image" in item:
                        pil_images.append(item["image"])
                    elif item.get("type") == "video" and "video" in item:
                        video_frames_list.extend(item["video"])
            videos_arg = [video_frames_list] if video_frames_list else None

            inputs = self.processor(text=text_prompt, images=pil_images or None, videos=videos_arg, return_tensors="pt")
            model_device = get_model_input_device(self.model)
            model_inputs = {
                k: (v.to(model_device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }

            stop_tokens = [self.tokenizer.eos_token_id]
            if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
                stop_tokens.append(self.tokenizer.eot_id)

            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
                "eos_token_id": stop_tokens,
                "pad_token_id": pad_id,
            }

            if num_beams > 1:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})

            print("[aistudynow] process(): calling model.generate()")
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            input_len = model_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

            print(f"[aistudynow] process(): done in {time.time() - start_time:.2f}s")
            final_text = text.strip()
            pbar.update_absolute(3, 3, None)
            return {"ui": {"text": [final_text]}, "result": (final_text,)}

        except Exception as e:
            print("[aistudynow] process(): ERROR")
            traceback.print_exc()
            msg = f"ERROR: {e}"
            return {"ui": {"text": [msg]}, "result": (msg,)}
        finally:
            if not keep_model_loaded:
                self.clear_model_resources()


class aistudynow_QwenVL(aistudynow_QwenVL_Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        base = aistudynow_QwenVL_Advanced.INPUT_TYPES()
        for key in [
            "temperature",
            "top_p",
            "num_beams",
            "repetition_penalty",
            "frame_count",
            "device",
            "use_torch_compile",
        ]:
            base["required"].pop(key, None)
        return base

    FUNCTION = "process_standard"

    def process_standard(
        self,
        model_name,
        quantization,
        preset_prompt,
        max_tokens,
        seed,
        attention_mode,
        keep_model_loaded, # Must match arg order in base
        custom_prompt="",
        image=None,
        video=None,
    ):
        return self.process(
            model_name=model_name,
            quantization=quantization,
            preset_prompt=preset_prompt,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            num_beams=1,
            frame_count=16,
            device="auto",
            use_torch_compile=False,
            attention_mode=attention_mode,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            keep_model_loaded=keep_model_loaded,
            seed=seed,
        )


class aistudynow_SaveText:
    CATEGORY = "🧠aistudynow/Utility"
    RETURN_TYPES = ()
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
                "filename": ("STRING", {"default": "qwenvl_output.txt"}),
            }
        }

    def save(self, text, filename):
        try:
            out_dir = Path(folder_paths.get_output_directory())
            out_path = out_dir / filename
            out_path.write_text(text if isinstance(text, str) else str(text), encoding="utf-8")
            print(f"[aistudynow] Saved text to: {out_path}")
        except Exception as e:
            print(f"[aistudynow] SaveText error: {e}")
        return tuple()


NODE_CLASS_MAPPINGS = {
    "aistudynow_QwenVL": aistudynow_QwenVL,
    "aistudynow_QwenVL_Advanced": aistudynow_QwenVL_Advanced,
    "aistudynow_SaveText": aistudynow_SaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "aistudynow_QwenVL": "QwenVL",
    "aistudynow_QwenVL_Advanced": "QwenVL (Advanced)",
    "aistudynow_SaveText": "aistudynow Save Text",
}
