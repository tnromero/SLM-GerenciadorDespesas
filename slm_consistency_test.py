from collections import Counter

from slm_gerenciadordespesas.config import OllamaConfig
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier


classifier = OllamaClassifier(
    OllamaConfig(
        model="qwen3:1.7b",
        temperature=0.0,
    )
)

descriptions = [
    "Abasteci o carro",
    "Comprei comida japonesa",
    "Assinei Disney Plus",
    "Paguei a conta de água",
    "Comprei um remédio",
    "Fiz uma doação",
]

runs = 5

for description in descriptions:
    results = []

    for _ in range(runs):
        result = classifier.classify(description)
        results.append(result.category.value)

    counts = Counter(results)

    print()
    print(description)
    print(f"Resultados: {results}")
    print(f"Consistência: {len(counts) == 1}")
    print(f"Distribuição: {dict(counts)}")
