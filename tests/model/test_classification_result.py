from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.classification_result import ClassificationResult
from slm_gerenciadordespesas.model.expense import Expense

def test_classification_result():
    expense = Expense(
        description="Netflix",
        category=Category.STREAMING,
    )
    source = "Ollama"
    confidence = 0.95

    result = ClassificationResult(
        expense=expense,
        source=source,
        confidence=confidence,
    )

    assert result.expense == expense
    assert result.source == source
    assert result.confidence == confidence