from dataclasses import dataclass

from slm_gerenciadordespesas.model.expense import Expense

@dataclass(frozen=True)
class ClassificationResult:
    expense: Expense
    source: str
    confidence: float