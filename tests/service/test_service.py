
from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.expense import Expense
from slm_gerenciadordespesas.service.expense_service import ExpenseService


class FakeClassifier(ExpenseClassifier):
    def classify(self, description: str) -> Expense:
        return Expense(
            description=description,
            category=Category.ALIMENTACAO,
        )


def test_service_uses_classifier():
    service = ExpenseService(FakeClassifier())

    result = service.classify("Despesa qualquer")

    assert result.description == "Despesa qualquer"
    assert result.category == Category.ALIMENTACAO