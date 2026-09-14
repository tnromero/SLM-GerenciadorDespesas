import csv
import time

from slm_gerenciadordespesas.controller.hybrid_classifier import HybridClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier

from slm_gerenciadordespesas.config import OllamaConfig


def load_dataset():
    with open("data/evaluation.csv", encoding="utf-8") as file:
        return list(csv.DictReader(file))


dataset = load_dataset()

classifier = HybridClassifier(
    rule_classifier=RuleBasedClassifier(),
    slm_classifier=OllamaClassifier(
        OllamaConfig(model="qwen3:1.7b")
    ),
)

correct = 0
total_time = 0

for item in dataset:
    description = item["description"]
    expected = item["expected_category"]

    start = time.perf_counter()

    result = classifier.classify(description)

    elapsed = time.perf_counter() - start
    total_time += elapsed

    predicted = result.category.value
    ok = predicted == expected

    if ok:
        correct += 1

    print(
        f"{'OK' if ok else 'ERRO':5} "
        f"{description:35} "
        f"esperado={expected:15} "
        f"obtido={predicted:15} "
        f"{elapsed:.3f}s"
    )


total = len(dataset)

print()
print("=" * 60)
print("HYBRID")
print("=" * 60)
print(f"Acurácia:     {correct / total * 100:.2f}%")
print(f"Acertos:      {correct}/{total}")
print(f"Tempo total:  {total_time:.2f}s")
print(f"Tempo médio:  {total_time / total:.3f}s")
print()
print(f"Total:        {classifier.total_calls}")
print(f"Chamadas SLM: {classifier.slm_calls}")
print(f"Taxa SLM:     {classifier.slm_call_rate:.2%}")