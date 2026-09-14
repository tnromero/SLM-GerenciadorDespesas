

from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier
from slm_gerenciadordespesas.model.classification_result import ClassificationResult
from slm_gerenciadordespesas.model.expense import Expense


class HybridClassifier(ExpenseClassifier):
    def __init__(
        self,
        rule_classifier: RuleBasedClassifier,
        slm_classifier: OllamaClassifier,
    ):
        self.rule_classifier = rule_classifier
        self.slm_classifier = slm_classifier

        self.total_calls = 0
        self.rule_calls = 0
        self.slm_calls = 0

    def classify(self, description: str) -> Expense:
        return self.classify_with_metadata(description).expense
    
    def classify_with_metadata(self, description: str) -> ClassificationResult:
        self.total_calls += 1

        try:
            result = self.rule_classifier.classify(description)
            self.rule_calls += 1
            return ClassificationResult(
                expense=result,
                source="rule",
                confidence=1.0
            )
        except ValueError:
            self.slm_calls += 1
            result = self.slm_classifier.classify(description)

            return ClassificationResult(
                expense=result,
                source="slm",
                confidence=0.8,
            )

    @property
    def rule_call_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0

        return self.rule_calls / self.total_calls

    @property
    def slm_call_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0

        return self.slm_calls / self.total_calls
