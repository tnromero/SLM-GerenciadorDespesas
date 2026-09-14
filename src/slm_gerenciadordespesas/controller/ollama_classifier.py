import ollama
from pydantic import ValidationError

from slm_gerenciadordespesas.config import OllamaConfig
from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.expense import Expense
from slm_gerenciadordespesas.model.inference_metrics import InferenceMetrics

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
        self.last_metrics: InferenceMetrics | None = None

    def classify(self, description: str) -> Expense:
        categories = ", ".join(
            category.value for category in Category
        )

        system_prompt = f"""
Você é um classificador de despesas.

Classifique a despesa em exatamente uma das categorias abaixo:

{categories}

Não crie novas categorias.
Retorne apenas uma categoria válida.

{FEW_SHOT_EXAMPLES}
        """

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
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": description,
                        },
                    ],
                    format=Expense.model_json_schema(),
                )

                self.last_metrics = InferenceMetrics(
                    total_duration_ns=response.get("total_duration", 0),
                    load_duration_ns=response.get("load_duration", 0),
                    prompt_eval_count=response.get("prompt_eval_count", 0),
                    eval_count=response.get("eval_count", 0),
                    prompt_eval_duration_ns=response.get("prompt_eval_duration", 0),
                    eval_duration_ns=response.get("eval_duration", 0),
                )
                return Expense.model_validate_json(
                    response.message.content or "{}"
                )

            except (ValidationError, ValueError) as exc:
                last_error = exc

            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"Falha ao classificar despesa após "
                f"{self.max_retries + 1} tentativas"
            ) from last_error

        raise RuntimeError(
            f"Falha ao classificar '{description}' "
            f"após {self.max_retries + 1} tentativas"
        ) from last_error