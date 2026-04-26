from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import torch
import gc

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


@dataclass
class GeneratedImage:
    path: str
    seed: int


class StableDiffusionGenerator:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        enable_xformers: bool = True,
        enable_attention_slicing: bool = False,
        width: int = 512,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
    ):
        self.model_id = model_id
        self.device = device
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        self._load_model(torch_dtype=torch_dtype, enable_xformers=enable_xformers, enable_attention_slicing=enable_attention_slicing)

    def _load_model(self, torch_dtype: torch.dtype, enable_xformers: bool = True, enable_attention_slicing: bool = False) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to(self.device)

        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()

    def ensure_model_loaded(self):
        if self.pipe is None:
            self._load_model(torch_dtype=torch.float16, enable_xformers=True, enable_attention_slicing=False)

    def generate(
        self,
        prompt: str,
        num_images: int,
        seed: int,
        output_dir: str,
        batch_size: int = 5,
        save_images: bool = True,
    ) -> List[GeneratedImage]:
        
        self.ensure_model_loaded()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results: List[GeneratedImage] = []
        remaining = num_images
        batch_index = 0

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            current_seed = seed + batch_index
            generator = torch.Generator(device=self.device).manual_seed(current_seed)

            with torch.inference_mode():
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=current_batch_size,
                ).images

            for j, image in enumerate(images):
                image_index = len(results)
                image_path = str(
                    Path(output_dir) / f"synthetic_{image_index:03d}_seed_{current_seed}.png"
                )

                if save_images:
                    image.save(image_path)

                results.append(
                    GeneratedImage(
                        path=image_path,
                        seed=current_seed,
                    )
                )
            del images
            self.cleanup_gpu_memory()

            remaining -= current_batch_size
            batch_index += 1

        return results
    
    def cleanup_gpu_memory(self) -> None:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def unload_model(self) -> None:

        if hasattr(self, "pipe") and self.pipe is not None:
            try:
                if torch.cuda.is_available():
                    self.pipe.to(dytpe=torch.float32)  # Convertir a float32 para liberar VRAM
                
                self.pipe.cpu()  # mover a CPU
            except Exception:
                pass

            del self.pipe
            self.pipe = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()