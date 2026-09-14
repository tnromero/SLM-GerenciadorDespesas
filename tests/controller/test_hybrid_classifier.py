from slm_gerenciadordespesas.controller.hybrid_classifier import HybridClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier
from slm_gerenciadordespesas.model.category import Category


def test_hybrid_classify_with_metadata_rule():
    classifier = HybridClassifier(
        rule_classifier=RuleBasedClassifier(),
        slm_classifier=OllamaClassifier(),
    )

    result = classifier.classify_with_metadata("Netflix")

    assert result.expense.category == Category.STREAMING
    assert result.source == "rule"
    assert result.confidence == 1.0

def test_hybrid_classify_with_metadata_slm():
    classifier = HybridClassifier(
        rule_classifier=RuleBasedClassifier(),
        slm_classifier=OllamaClassifier(),
    )

    result = classifier.classify_with_metadata("Assinei Disney Plus")

    assert result.expense.category == Category.STREAMING
    assert result.source == "slm"
    assert result.confidence == 0.8