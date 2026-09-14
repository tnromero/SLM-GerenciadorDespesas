import time

from slm_gerenciadordespesas.controller.ollama_classifier import (
    OllamaClassifier
)
from slm_gerenciadordespesas.controller.ruled_based_classifier import (
    RuleBasedClassifier
)


expenses = [
    ("Abasteci o carro", "Carro"),
    ("Gasolina do carro", "Carro"),
    ("Almoço com a família", "Alimentação"),
    ("Comida japonesa no restaurante", "Alimentação"),
    ("Pedi comida pelo iFood", "iFood"),
    ("Conta de energia da casa", "Contas fixas"),
    ("Mensalidade da Netflix", "Streaming"),
    ("Remédio para gripe", "Saúde"),
    ("Consulta médica", "Saúde"),
    ("Pagamento do IPVA", "Imposto"),
    ("Seguro do T-Cross", "Carro"),
    ("Doação para a AACD", "Doação"),
    ("Mensalidade do condomínio", "Contas fixas"),
    ("Plano de internet de casa", "Contas fixas"),
]


def benchmark(name, classifier):
    correct = 0
    total_time = 0

    print(f"\n=== {name} ===")

    for description, expected in expenses:
        start = time.perf_counter()

        try:
            result = classifier.classify(description)
        except ValueError as e:
            print(f"{description:25} | {'DESCONHECIDA':15} | {'ERRO':5} | {time.perf_counter() - start:.3f}s")
            continue

        elapsed = time.perf_counter() - start
        total_time += elapsed

        predicted = result.category.value
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(
            f"{description:25} | "
            f"{predicted:15} | "
            f"{'OK' if is_correct else 'ERRO':5} | "
            f"{elapsed:.3f}s"
        )

    accuracy = correct / len(expenses) * 100
    average_time = total_time / len(expenses)

    print(f"\nAcurácia: {accuracy:.1f}%")
    print(f"Tempo total: {total_time:.3f}s")
    print(f"Tempo médio: {average_time:.3f}s")


# benchmark(
#     "RULE BASED",
#     RuleBasedClassifier(),
# )
# 
# benchmark(
#     "QWEN 1.7B",
#     OllamaClassifier(),
# )

models = [
    "qwen3:1.7b",
    "qwen3:4b",
]

for model in models:
    benchmark(
        f"QWEN {model}",
        OllamaClassifier(model),
    )