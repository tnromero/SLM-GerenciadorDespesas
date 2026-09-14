import ollama

from slm_gerenciadordespesas.config import OllamaConfig
from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.expense import Expense


FEW_SHOT_EXAMPLES = """
Exemplos de classificação:

"Combustível" -> Carro
"Abasteci o carro" -> Carro
"Netflix" -> Streaming
"IPVA" -> Imposto
"Farmácia" -> Saúde
"iFood" -> iFood
"Almoço em restaurante" -> Alimentação
"Condomínio" -> Contas fixas
"AACD" -> Doação
"""

class OllamaClassifier(ExpenseClassifier):
    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()

    def classify(self, description: str) -> Expense:
        categories = ", ".join(category.value for category in Category)

        response = ollama.chat(
            model=self.config.model,
            think=self.config.think,
            options={"temperature": self.config.temperature},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um classificador de despesas.\n\n"
                        f"{FEW_SHOT_EXAMPLES}\n"
                        "Classifique a próxima despesa em exatamente uma "
                        f"das seguintes categorias: {categories}.\n"
                        "Não crie novas categorias."
                    ),
                },
                {
                    "role": "user",
                    "content": description,
                },
            ],
            format=Expense.model_json_schema(),
        )

        return Expense.model_validate_json(response.message.content)