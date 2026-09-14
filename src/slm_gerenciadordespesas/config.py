from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "qwen3:1.7b"
    temperature: float = 0.0
    think: bool = False
