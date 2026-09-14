from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier


class ExpenseService:
    def __init__(self, classifier: ExpenseClassifier):
        self.classifier = classifier

    def classify(self, description: str):
        return self.classifier.classify(description)