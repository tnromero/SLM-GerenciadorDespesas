from abc import ABC, abstractmethod

from slm_gerenciadordespesas.model.expense import Expense


class ExpenseClassifier(ABC):
    @abstractmethod
    def classify(self, description: str) -> Expense:
        ...