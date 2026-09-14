

from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier
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
        self.slm_calls = 0

    def classify(self, description: str) -> Expense:
        self.total_calls += 1

        try:
            return self.rule_classifier.classify(description)
        except ValueError:
            self.slm_calls += 1
            return self.slm_classifier.classify(description)

    @property
    def slm_call_rate(self) -> float:
        return self.slm_calls / self.total_calls