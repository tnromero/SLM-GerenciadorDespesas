

from pydantic import ValidationError
import pytest

from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.expense import Expense


def test_expense_accepts_valid_category():
    expense = Expense.model_validate(
        {
            "description": "Netflix",
            "category": "Streaming",
        }
    )

    assert expense.category == Category.STREAMING


def test_expense_rejects_invalid_category():
    with pytest.raises(ValidationError):
        Expense.model_validate(
            {
                "description": "Netflix",
                "category": "Cinema",
            }
        )