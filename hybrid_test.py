

from slm_gerenciadordespesas.controller.hybrid_classifier import HybridClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier


classifier = HybridClassifier(
    RuleBasedClassifier(),
    OllamaClassifier(),
)

expenses = [
    "Netflix",
    "IPVA",
    "Abasteci o carro",
    "Assinei Disney Plus",
    "Farmácia",
    "Comprei um lanche no McDonalds",
    "Condomínio",
]

for description in expenses:
    result = classifier.classify(description)

    print(
        f"{description:35} -> "
        f"{result.category.value}"
    )

print()
print(f"Total: {classifier.total_calls}")
print(f"Chamadas SLM: {classifier.slm_calls}")
print(f"Taxa SLM: {classifier.slm_call_rate:.1%}")
