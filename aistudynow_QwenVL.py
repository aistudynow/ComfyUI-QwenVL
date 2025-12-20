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

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download, HfApi, hf_hub_download
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import folder_paths

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

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


def get_model_info(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name, {})


def get_device_info() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {
            "available": True,
            "total_memory": total_mem,
            "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3),
        }
    else:
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {
        "total": sys_mem.total / 1024**3,
        "available": sys_mem.available / 1024**3,
    }

    device_info = {
        "gpu": gpu_info,
        "system_memory": sys_mem_info,
        "device_type": "cpu",
        "recommended_device": "cpu",
        "memory_sufficient": True,
        "warning_message": "",
    }

    if platform.system() == "Darwin" and platform.processor() == "arm":
        device_info.update({"device_type": "apple_silicon", "recommended_device": "mps"})
        if sys_mem_info["total"] < 16:
            device_info.update(
                {
                    "memory_sufficient": False,
                    "warning_message": "Apple Silicon memory is less than 16GB, performance may be affected.",
                }
            )
    elif gpu_info["available"]:
        device_info.update({"device_type": "nvidia_gpu", "recommended_device": "cuda"})
        if gpu_info["total_memory"] < 8:
            device_info.update(
                {
                    "memory_sufficient": False,
                    "warning_message": "GPU VRAM is less than 8GB, performance may be degraded.",
                }
            )

    return device_info


def check_memory_requirements(model_name: str, quantization: str, device_info: dict) -> str:
    model_info = get_model_info(model_name)
    vram_req = model_info.get("vram_requirement", {})
    quant_map = {
        Quantization.Q4_BIT: vram_req.get("4bit", 0.0),
        Quantization.Q8_BIT: vram_req.get("8bit", 0.0),
        Quantization.NONE: vram_req.get("full", 0.0),
    }

    base_memory = quant_map.get(quantization, 0.0)
    device = device_info["recommended_device"]
    use_cpu_mps = device in ["cpu", "mps"]

    required_mem = base_memory * (1.5 if use_cpu_mps else 1.0)
    available_mem = (
        device_info["system_memory"]["available"] if use_cpu_mps else device_info["gpu"]["free_memory"]
    )
    mem_type = "System RAM" if use_cpu_mps else "GPU VRAM"

    if required_mem * 1.2 > available_mem:
        print(f"Warning: Insufficient {mem_type} ({available_mem:.2f}GB available). Lowering quantization...")
        if quantization == Quantization.NONE:
            return Quantization.Q8_BIT
        if quantization == Quantization.Q8_BIT:
            return Quantization.Q4_BIT
        raise RuntimeError(f"Insufficient {mem_type} even for 4-bit quantization.")
    return quantization


def check_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    except Exception:
        return False
    return False


def _human_bytes(n_bytes: int) -> str:
    if n_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n_bytes)
    for u in units:
        if s < 1024 or u == "TB":
            return f"{s:.2f} {u}"
        s /= 1024.0


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


class ImageProcessor:
    def to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)


class ModelDownloader:
    def __init__(self, configs):
        self.configs = configs
        self.models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        self.models_dir.mkdir(parents=True, exist_ok=True)
		
		
		
		
		
		
		
		
		
    @staticmethod
    def _infer_model_type(repo_id: str) -> str | None:
        repo_lower = repo_id.lower()

        if "qwen3-vl" in repo_lower:
            return "qwen3_vl"
        if "qwen2.5-vl" in repo_lower:
            return "qwen2_5_vl"
        return None

    def _ensure_model_type(self, model_path: Path, repo_id: str):
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

        inferred_type = self._infer_model_type(repo_id)
        if inferred_type is None:
            print(f"[aistudynow] Warning: Unable to infer model_type for {repo_id}. Please update config.json manually.")
            return

        config_data["model_type"] = inferred_type
        try:
            config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
            print(f"[aistudynow] Added missing model_type='{inferred_type}' to {config_file}")
        except Exception as e:
            print(f"[aistudynow] Warning: failed to update config.json for {repo_id}: {e}")
		
		
		
		
		
		
		
		
		
		
		
		

    def _print_transfer_hint(self):
        if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").strip():
            print("[aistudynow] Tip: enable faster downloads by installing hf_transfer and setting HF_HUB_ENABLE_HF_TRANSFER=1")

    def ensure_model_available(self, model_name: str) -> str:
        model_info = self.configs.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in configuration.")

        repo_id = model_info["repo_id"]
        model_folder_name = repo_id.split("/")[-1]
        model_path = self.models_dir / model_folder_name
        model_path.mkdir(parents=True, exist_ok=True)

        if _has_required_weights(model_path):
            print(f"[aistudynow] Model '{model_name}' ready at {model_path}.")
            return str(model_path)

        self._print_transfer_hint()

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
            bar = tqdm(total=total_known, unit="B", unit_scale=True, desc="[aistudynow] Total", leave=False) if use_bar else None

            for i, s in enumerate(files_to_get, 1):
                local_file = model_path / s.rfilename
                size = getattr(s, "size", None)
                size_txt = _human_bytes(size)

                if local_file.exists() and (size is None or local_file.stat().st_size == size):
                    if bar and size:
                        bar.update(size)
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
                        bar.update(size)
                except Exception as e:
                    print(f"[aistudynow]   failed: {e}")

            if bar:
                bar.close()
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
			
			self._ensure_model_type(model_path, repo_id)

        print(f"[aistudynow] Model '{model_name}' ready at {model_path}.")
        return str(model_path)


class aistudynow_QwenVL_Advanced:
    CATEGORY = "🧠aistudynow/QwenVL"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()

        print(f"QwenVL Node Initialized. Device: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"Warning: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        if self.model is not None:
            print("Releasing model resources...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto"):
        effective_device = self.device_info["recommended_device"] if device == "auto" else device

        if (
            self.model is not None
            and self.current_model_name == model_name
            and self.current_quantization == quantization_str
            and self.current_device == effective_device
        ):
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 models require a GPU with Compute Capability 8.9 or higher (for example RTX 4090). "
                        f"Your GPU capability is {cc}. Select a non FP8 model."
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)

        # choose dtype
        is_prequant = get_model_info(model_name).get("quantized", False)
        quant_config, load_dtype = None, (None if is_prequant else torch.float16)

        if not is_prequant:
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

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}

        load_kwargs = {
            "device_map": device_map,
            "dtype": load_dtype,
            "attn_implementation": "flash_attention_2" if check_flash_attention() else "sdpa",
            "use_safetensors": True,
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"Loading model '{model_name}'...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            **load_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.current_model_name = model_name
        self.current_quantization = quantization_str
        self.current_device = effective_device
        print("Model loaded successfully.")

    @classmethod
    def INPUT_TYPES(cls):
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_")]
        default_model = next(
            (name for name in model_names if MODEL_CONFIGS[name].get("default")), model_names[0] if model_names else ""
        )
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])

        return {
            "required": {
                "model_name": (model_names, {"default": default_model}),
                "quantization": (list(Quantization.get_values()), {"default": Quantization.Q8_BIT}),
                "preset_prompt": (preset_prompts, {"default": preset_prompts[0]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Custom prompt"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
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
        seed,
        custom_prompt="",
        image=None,
        video=None,
        keep_model_loaded=True,
    ):
        start_time = time.time()
        try:
            print("[aistudynow] process(): start")
            torch.manual_seed(seed)

            print(f"[aistudynow] process(): load_model(model={model_name}, quant={quantization}, device={device})")
            self.load_model(model_name, quantization, device)
            effective_device = self.current_device
            print(f"[aistudynow] process(): device resolved -> {effective_device}")

            final_prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else preset_prompt
            print(f"[aistudynow] process(): final_prompt='{final_prompt[:80]}'")

            conversation = [{"role": "user", "content": []}]

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

            pil_images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
            video_frames_list = [frm for item in conversation[0]["content"] if item["type"] == "video" for frm in item["video"]]
            videos_arg = [video_frames_list] if video_frames_list else None

            inputs = self.processor(text=text_prompt, images=pil_images or None, videos=videos_arg, return_tensors="pt")
            model_inputs = {k: v.to(effective_device) for k, v in inputs.items() if torch.is_tensor(v)}

            stop_tokens = [self.tokenizer.eos_token_id]
            if hasattr(self.tokenizer, "eot_id"):
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
            input_len = model_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

            print(f"[aistudynow] process(): done in {time.time() - start_time:.2f}s")
            return (text.strip(),)

        except Exception as e:
            print("[aistudynow] process(): ERROR")
            traceback.print_exc()
            msg = f"ERROR: {e}"
            return (msg,)
        finally:
            if not keep_model_loaded:
                self.clear_model_resources()


class aistudynow_QwenVL(aistudynow_QwenVL_Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        base = aistudynow_QwenVL_Advanced.INPUT_TYPES()
        for key in ["temperature", "top_p", "num_beams", "repetition_penalty", "frame_count", "device"]:
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
        custom_prompt="",
        image=None,
        video=None,
        keep_model_loaded=True,
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
