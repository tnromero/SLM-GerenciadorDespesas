from pydantic import BaseModel

from slm_gerenciadordespesas.model.category import Category

class Expense(BaseModel):
    description: str
    category: Category