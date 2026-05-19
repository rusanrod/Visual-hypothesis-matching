    
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptPack:
    raw_command: str
    object_phrase: str
    positive_prompt: str
    negative_prompt: str


class PromptBuilder:
    def __init__(self):
        self.base_style = (
            "isolated product photo, single object, centered composition, "
            "full object visible, clean white background, studio lighting, "
            "soft shadow, sharp focus, high detail, realistic, front view"
        )

        self.quality_terms = (
            "well lit, clear shape, visible color, visible texture, "
            "no occlusion, no cropping, no clutter"
        )

        self.negative_prompt = (
            "cropped object, partially visible object, occluded object, "
            "multiple unrelated objects, cluttered background, complex scene, "
            "hands, people, person, text, watermark, logo, label, blurry, "
            "low resolution, distorted shape, deformed object, bad perspective, "
            "dark lighting, overexposed, underexposed, reflection glare"
        )

        self.article_prefixes = (
            "a ",
            "an ",
            "the ",
            "one ",
            "a pair of ",
            "pair of ",
        )

    def build(self, raw_command: str) -> PromptPack:
        object_phrase = self._normalize_object_phrase(raw_command)

        positive_prompt = (
            f"{self.base_style}, {object_phrase}, {self.quality_terms}"
        )

        return PromptPack(
            raw_command=raw_command,
            object_phrase=object_phrase,
            positive_prompt=positive_prompt,
            negative_prompt=self.negative_prompt,
        )

    def build_variants(self, raw_command: str, n: int = 10) -> list[PromptPack]:
        object_phrase = self._normalize_object_phrase(raw_command)

        view_variants = [
            "front view",
            "slightly angled front view",
            "three quarter view",
            "top front view",
            "low angle product view",
        ]

        lighting_variants = [
            "soft studio lighting",
            "diffuse lighting",
            "bright even lighting",
            "natural soft light",
            "controlled product photography lighting",
        ]

        prompts = []

        for i in range(n):
            view = view_variants[i % len(view_variants)]
            lighting = lighting_variants[i % len(lighting_variants)]

            positive = (
                f"isolated product photo of {object_phrase}, "
                f"single object, centered composition, full object visible, "
                f"clean white background, {lighting}, {view}, "
                f"sharp focus, high detail, realistic, visible color, "
                f"visible texture, no occlusion, no cropping"
            )

            prompts.append(
                PromptPack(
                    raw_command=raw_command,
                    object_phrase=object_phrase,
                    positive_prompt=positive,
                    negative_prompt=self.negative_prompt,
                )
            )

        return prompts

    def _normalize_object_phrase(self, text: str) -> str:
        text = text.strip().lower()
        text = text.replace('"', "").replace("'", "")

        for prefix in sorted(self.article_prefixes, key=len, reverse=True):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return text