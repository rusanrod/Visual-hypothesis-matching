from typing import List


class PromptBuilder:
    def __init__(self, base_style: str = ""):
        self.base_style = base_style

    def build_prompt(
        self,
        raw_prompt: str,
        object_name: str = "",
        features: List[str] | None = None,
    ) -> str:
        features = features or []

        parts = []

        if raw_prompt:
            parts.append(raw_prompt)

        if object_name:
            parts.append(f"target object: {object_name}")

        if features:
            parts.append("features: " + ", ".join(features))

        if self.base_style:
            parts.append(self.base_style)

        return ". ".join(parts)