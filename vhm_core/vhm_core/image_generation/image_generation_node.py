
import torch

import rclpy
from rclpy.node import Node

import random
from pathlib import Path
import json
import time

from vhm_interfaces.srv import GenerateReferences #type: ignore

from vhm_core.image_generation.stable_diffusion_generator import StableDiffusionGenerator
from vhm_core.image_generation.prompt_builder import PromptBuilder


class ImageGenerationNode(Node):
    def __init__(self):
        super().__init__("image_generation_node")

        params = [
                ("model_id", "runwayml/stable-diffusion-v1-5"),
                ("device", "cuda"),
                ("dtype", "float16"),
                ("enable_xformers", True),
                ("enable_attention_slicing", False),
                ("width", 384),
                ("height", 384),
                ("steps", 25),
                ("guidance_scale", 7.5),
                ("negative_prompt", "blurry, low quality, distorted, deformed"),
                ("base_style", "single object, centered, clean background, realistic photo"),
            ],

        for name, default in params:
            self.declare_parameter(name, default)

        self.default_output_dir = Path.joinpath(Path.home(), "vhm_ws", "src", "vhm_results", "generated_references")

        self.prompt_builder = PromptBuilder(
            base_style=self.get_parameter("base_style").get_parameter_value().string_value
        )

        self.get_logger().info("Cargando Stable Diffusion...")

        self.generator = StableDiffusionGenerator(
            model_id=self.get_parameter("model_id").get_parameter_value().string_value,
            device=self.get_parameter("device").get_parameter_value().string_value,
            dtype=self.get_parameter("dtype").get_parameter_value().string_value,
            enable_xformers=self.get_parameter("enable_xformers").get_parameter_value().bool_value,
            enable_attention_slicing=self.get_parameter("enable_attention_slicing").get_parameter_value().bool_value,
            width=self.get_parameter("width").get_parameter_value().integer_value,
            height=self.get_parameter("height").get_parameter_value().integer_value,
            steps=self.get_parameter("steps").get_parameter_value().integer_value,
            guidance_scale=self.get_parameter("guidance_scale").get_parameter_value().double_value,
            negative_prompt=self.get_parameter("negative_prompt").get_parameter_value().string_value,
        )

        self.img_gen_srv = self.create_service(
            GenerateReferences,
            "vhm_core/generate_references",
            self.generate_callback,
        )

        self.get_logger().info("Nodo image_generation listo.")

    def _build_output_dir(self, experiment_id: str) -> str:
        bank_id = experiment_id.strip() if experiment_id else "test"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.default_output_dir) / bank_id / timestamp

        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)


    def _save_reference_bank(
        self,
        experiment_id: str,
        prompt: str,
        seed: int,
        num_images_requested: int,
        image_paths: list[str],
        batch_size: int,
        start_time: float,
        end_time: float,
    ) -> str:

        bank_id = experiment_id.strip() if experiment_id else "test"

        output_dir = Path(image_paths[0]).parent if image_paths else Path(self.default_output_dir) / bank_id
        bank_path = output_dir / "reference_bank.json"

        total_time = end_time - start_time
        num_generated = len(image_paths)
        avg_time = total_time / num_generated if num_generated > 0 else 0.0

        images_metadata = []
        for idx, path in enumerate(image_paths):
            filename = Path(path).name
            images_metadata.append({
                "index": idx,
                "filename": filename,
                "path": path,
                "seed": seed,              # seed base del experimento
                "batch_index": idx // batch_size
            })

        gpu_metadata = self._get_gpu_metadata()

        payload = {
            "reference_bank_id": bank_id,
            "prompt": prompt,

            "num_images_requested": num_images_requested,
            "num_images_generated": num_generated,

            "seed": seed,
            "batch_size": batch_size,

            # --- configuración del modelo ---
            "model_id": self.generator.model_id,
            "scheduler": type(self.generator.pipe.scheduler).__name__, #type: ignore
            "width": self.generator.width,
            "height": self.generator.height,
            "steps": self.generator.steps,
            "guidance_scale": self.generator.guidance_scale,
            "negative_prompt": self.generator.negative_prompt,

            # --- métricas ---
            "generation_started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "generation_finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "total_generation_time_sec": round(total_time, 4),
            "avg_time_per_image_sec": round(avg_time, 4),

            # --- entorno ---
            "gpu": gpu_metadata,
            "device": self.generator.device,
            "torch_dtype": str(self.generator.pipe.unet.dtype), #type: ignore

            # --- outputs ---
            "output_dir": str(output_dir),

            "images": images_metadata
        }

        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(bank_path)
    
    def _get_gpu_metadata(self) -> dict:

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu_name": "",
                "vram_allocated_mb": 0.0,
                "vram_reserved_mb": 0.0,
                "vram_max_allocated_mb": 0.0,
                "vram_total_mb": 0.0,
            }

        device_idx = torch.cuda.current_device()

        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(device_idx),
            "gpu_index": device_idx,
            "vram_allocated_mb": round(torch.cuda.memory_allocated(device_idx) / 1024**2, 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(device_idx) / 1024**2, 2),
            "vram_max_allocated_mb": round(torch.cuda.max_memory_allocated(device_idx) / 1024**2, 2),
            "vram_total_mb": round(torch.cuda.get_device_properties(device_idx).total_memory / 1024**2, 2),
        }

    def generate_callback(self, request, response):
        try:
            num_images = request.num_images if request.num_images > 0 else 10
            seed = request.seed if request.seed > 0 else random.randint(0, 1000)
            batch_size = 5

            output_dir = self._build_output_dir(request.experiment_id)

            self.get_logger().info(f"Prompt: {request.prompt}")
            self.get_logger().info(f"Generating {num_images} reference images...")

            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            generated = self.generator.generate(
                prompt=request.prompt,
                num_images=num_images,
                seed=seed,
                output_dir=output_dir,
                save_images=request.save_reference_bank,
                batch_size=batch_size
            )

            end_time = time.time()

            image_paths = [img.path for img in generated]

            if request.save_reference_bank:
                self._save_reference_bank(
                    experiment_id=request.experiment_id,
                    prompt=request.prompt,
                    seed=seed,
                    num_images_requested=num_images,
                    image_paths=image_paths,
                    batch_size=batch_size,
                    start_time=start_time,
                    end_time=end_time
                )

            response.success = True
            response.message = "References generated successfully."
            response.generated_reference_count = len(image_paths)
            response.output_dir = output_dir

            return response

        except Exception as e:
            self.get_logger().error(f"Error generating references: {e}")

            response.success = False
            response.message = str(e)
            response.generated_reference_count = 0
            response.output_dir = ""

            return response
        
        finally:
            self.generator.unload_model()


def main(args=None):
    rclpy.init(args=args)
    node = ImageGenerationNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()