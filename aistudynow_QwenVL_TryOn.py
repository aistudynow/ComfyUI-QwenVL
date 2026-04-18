import torch
import time
from typing import Any
from aistudynow_QwenVL import aistudynow_QwenVL_Advanced, get_device_options, ATTENTION_MODES, Quantization
from comfy.utils import ProgressBar

class aistudynow_QwenVL_TryOn(aistudynow_QwenVL_Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        required = base_inputs["required"].copy()
        
        if "custom_prompt" in required: del required["custom_prompt"]
        if "preset_prompt" in required: del required["preset_prompt"]
        if "image" in required: del required["image"]
        if "video" in required: del required["video"]
        
        new_required = {
            "model_name": required.pop("model_name"),
            "quantization": required.pop("quantization"),
            "person_description": ("STRING", {"default": "Describe short info of the person in this image", "multiline": True}),
            "top_description": ("STRING", {"default": "Describe the top outfit in this image", "multiline": True}),
            "bottom_description": ("STRING", {"default": "Describe the bottom outfit in this image", "multiline": True}),
        }
        new_required.update(required)
        
        return {
            "required": new_required,
            "optional": {
                "image_person": ("IMAGE",),
                "image_top": ("IMAGE",),
                "image_bottom": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process_tryon"
    CATEGORY = "🧠aistudynow/QwenVL"

    @torch.no_grad()
    def process_tryon(
        self,
        model_name,
        quantization,
        person_description,
        top_description,
        bottom_description,
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
        image_person=None,
        image_top=None,
        image_bottom=None,
    ):
        start_time = time.time()
        pbar = ProgressBar(3)
        try:
            print("[aistudynow TryOn] process(): start")
            torch.manual_seed(seed)
            pbar.update_absolute(1, 3, None)

            # 1. Load Model
            print(
                "[aistudynow TryOn] process(): load_model("
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
            pbar.update_absolute(2, 5, None)

            # Helper function to describe an image
            def describe_image(image_tensor, prompt, pbar_val):
                conversation: list[dict[str, Any]] = [{"role": "user", "content": []}]
                if image_tensor is not None:
                    conversation[0]["content"].append({"type": "image", "image": self.image_processor.to_pil(image_tensor)})
                conversation[0]["content"].append({"type": "text", "text": prompt})
                
                text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                pil_images = []
                for item in conversation[0]["content"]:
                    if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                        pil_images.append(item["image"])
                
                inputs = self.processor(
                    text=[text_prompt],
                    images=pil_images if pil_images else None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                
                model_device = next(self.model.parameters()).device
                inputs = inputs.to(model_device)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=(temperature > 0),
                    num_beams=num_beams,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                pbar.update_absolute(pbar_val, 5, None)
                return output_text[0].strip()

            # 2. Extract Descriptions
            print("[aistudynow TryOn] process(): Extracting person description")
            person_desc = person_description.strip()
            
            # Formating prompt to strictly enforce short answers
            prefix = "Describe this in a maximum of 10 words. DO NOT write full sentences. DO NOT include introductory text. Example format: 'young woman with long dark wavy hair'.\n\nYour description: "
            
            if image_person is not None:
                 p_prompt = prefix + "IGNORE ALL CLOTHING AND IGNORE POSE. ONLY describe the person's physical features (hair, facial features, age) in under 10 words. " + person_description.strip()
                 person_desc = describe_image(image_person, p_prompt, 3)

            print("[aistudynow TryOn] process(): Extracting top description")
            top_desc = top_description.strip()
            if image_top is not None:
                 t_prompt = prefix + top_description.strip()
                 top_desc = describe_image(image_top, t_prompt, 4)

            print("[aistudynow TryOn] process(): Extracting bottom description")
            bottom_desc = bottom_description.strip()
            if image_bottom is not None:
                 b_prompt = prefix + "IGNORE ALL UPPER BODY CLOTHING. DO NOT describe ANY shirts, tops, or jackets. ONLY describe the pants/skirt/bottom. " + bottom_description.strip()
                 bottom_desc = describe_image(image_bottom, b_prompt, 5)

            # 3. Build Final TryOn Prompt
            final_prompt = f"TRYON {person_desc}. Replace the outfit with {top_desc} and {bottom_desc} as shown in the reference images. The final image is a full body shot."
            print(f"[aistudynow TryOn] process(): final_prompt='{final_prompt}'")

            print(f"[aistudynow TryOn] process(): finished in {time.time() - start_time:.2f}s")
            return (final_prompt,)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[aistudynow] process(): OOM error: {e}")
            if hasattr(self, "model") and self.model is not None:
                self.model.to("cpu")
            torch.cuda.empty_cache()
            return ("Error: Out of Memory. Please try a smaller model or lower resolution image.",)
        except Exception as e:
            print(f"[aistudynow] process(): Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return (f"Error: {e}",)
        finally:
            if not keep_model_loaded:
                print("[aistudynow] process(): unloading model")
                self.clear()

NODE_CLASS_MAPPINGS = {
    "aistudynow_QwenVL_TryOn": aistudynow_QwenVL_TryOn
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "aistudynow_QwenVL_TryOn": "QwenVL TryOn"
}
