from slm_gerenciadordespesas.controller.ollama_classifier import (
    OllamaClassifier
)

classifier = OllamaClassifier()

expenses = [
    "Combustível",
    "Almoço em restaurante",
    "Netflix",
    "IPVA",
    "Farmácia",
    "Conta de água",
]

for description in expenses:
    result = classifier.classify(description)
    print(f"{description:25} -> {result.category}")