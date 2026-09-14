import csv
import sys
import time

from slm_gerenciadordespesas.config import OllamaConfig
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier


if len(sys.argv) > 1:
    DATASET_DEFAULT_PATH = sys.argv[1]
else:
    DATASET_DEFAULT_PATH = "data/evaluation.csv"

def load_dataset(path: str = DATASET_DEFAULT_PATH):
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evaluate(name, classifier, dataset):
    correct = 0
    total_time = 0

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    for item in dataset:
        description = item["description"]
        expected = item["expected_category"]

        start = time.perf_counter()
        try:
            result = classifier.classify(description)
        except ValueError as e:
            print(f"{description:35} | {'DESCONHECIDA':15} | {'ERRO':5} | {time.perf_counter() - start:.3f}s")
            continue
        elapsed = time.perf_counter() - start

        predicted = result.category.value
        total_time += elapsed

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
    accuracy = correct / total * 100
    average = total_time / total

    print()
    print(f"Acurácia:     {accuracy:.2f}%")
    print(f"Acertos:      {correct}/{total}")
    print(f"Tempo total:  {total_time:.2f}s")
    print(f"Tempo médio:  {average:.3f}s")


dataset = load_dataset()
print(dataset)
print(f"Total de despesas no dataset: {len(dataset)}")

evaluate(
    "RULE BASED",
    RuleBasedClassifier(),
    dataset,
)

evaluate(
    "QWEN 1.7B",
    OllamaClassifier(OllamaConfig(model="qwen3:1.7b")),
    dataset,
)

evaluate(
    "QWEN 4B",
    OllamaClassifier(OllamaConfig(model="qwen3:4b")),
    dataset,
)
