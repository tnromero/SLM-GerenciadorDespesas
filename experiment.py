from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.config import OllamaConfig


description = "Comprei um lanche no restaurante"


for temperature in [0.0, 0.3, 0.7]:
    config = OllamaConfig(
        model="qwen3:1.7b",
        temperature=temperature,
        think=False,
    )

    classifier = OllamaClassifier(config)

    result = classifier.classify(description)

    print(
        f"temperature={temperature} "
        f"-> {result.category.value}"
    )