import gc
from typing import List

import torch
import numpy as np
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        self.dtype = torch.float16 if dtype == "float16" and self.device == "cuda" else torch.float32

        self.model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()


    def encode_image_paths(self, image_paths: List[str]) -> torch.Tensor:
        images = [
            PILImage.open(path).convert("RGB")
            for path in image_paths
        ]

        return self.encode_pil_images(images)

    def encode_cv_images(self, cv_images: List[np.ndarray]) -> torch.Tensor:
        pil_images = []

        for image in cv_images:
            rgb = image[:, :, ::-1]
            pil_images.append(PILImage.fromarray(rgb).convert("RGB"))

        return self.encode_pil_images(pil_images)

    def encode_pil_images(self, images: List[PILImage.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0)

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            embeddings = self.model.get_image_features(**inputs)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.detach().cpu()

    def cleanup_gpu_memory(self):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()