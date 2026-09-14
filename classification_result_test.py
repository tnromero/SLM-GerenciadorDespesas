
from slm_gerenciadordespesas.controller.hybrid_classifier import HybridClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier



classifier = HybridClassifier(
    RuleBasedClassifier(),
    OllamaClassifier(),
)

for description in ['Netflix', 'Assinei Disney Plus']:
    result = classifier.classify_with_metadata(description)

    print(
        description,
        '| categoria=', result.expense.category.value,
        '| origem=', result.source,
        '| confiança=', result.confidence,
    )
