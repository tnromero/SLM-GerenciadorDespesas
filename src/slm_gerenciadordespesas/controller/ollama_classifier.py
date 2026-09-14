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
    def __init__(
        self,
        config: OllamaConfig | None = None,
        max_retries: int = 2,
    ):
        self.config = config or OllamaConfig()
        self.max_retries = max_retries

    def classify(self, description: str) -> Expense:
        categories = ", ".join(
            category.value for category in Category
        )

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.config.model,
                    think=self.config.think,
                    options={
                        "temperature": self.config.temperature,
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um classificador de despesas.\n"
                                f"Categorias válidas: {categories}\n"
                                "Escolha exatamente uma categoria. "
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

                return Expense.model_validate_json(
                    response.message.content
                )

            except Exception as error:
                last_error = error

                if attempt < self.max_retries:
                    continue

        raise RuntimeError(
            f"Falha ao classificar '{description}' "
            f"após {self.max_retries + 1} tentativas"
        ) from last_error