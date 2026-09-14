from slm_gerenciadordespesas.controller.classifier import ExpenseClassifier
from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.model.expense import Expense


class RuleBasedClassifier(ExpenseClassifier):
    RULES = {
        "combustível": Category.CARRO,
        "conectcar": Category.CARRO,
        "seguro do carro": Category.CARRO,
        "almoço": Category.ALIMENTACAO,
        "restaurante": Category.ALIMENTACAO,
        "mercado": Category.ALIMENTACAO,
        "ifood": Category.IFOOD,
        "energia elétrica": Category.CONTAS_FIXAS,
        "aluguel da garagem": Category.CONTAS_FIXAS,
        "condomínio": Category.CONTAS_FIXAS,
        "internet": Category.CONTAS_FIXAS,
        "iptu": Category.IMPOSTO,
        "ipva": Category.IMPOSTO,
        "netflix": Category.STREAMING,
        "hbo max": Category.STREAMING,
        "farmácia": Category.SAUDE,
        "aacd": Category.DOACAO,
    }

    def classify(self, description: str) -> Expense:
        normalized = description.strip().lower()

        for keyword, category in self.RULES.items():
            if keyword in normalized:
                return Expense(
                    description=description,
                    category=category,
                )

        raise ValueError(
            f"Não foi possível classificar a despesa: {description}"
        )


