
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier

c = OllamaClassifier()

for text in [
    "Comprei um lanche no McDonalds",
    "Abasteci o T-Cross",
    "Paguei a conta de água",
    "Comprei um remédio",
    "Assinei o Disney Plus",
]:
    result = c.classify(text)
    print(f"{text} -> {result.category.value}")
