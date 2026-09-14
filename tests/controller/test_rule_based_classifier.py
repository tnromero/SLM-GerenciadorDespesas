import pytest

from slm_gerenciadordespesas.model.category import Category
from slm_gerenciadordespesas.controller.ruled_based_classifier import RuleBasedClassifier


@pytest.fixture
def classifier():
    return RuleBasedClassifier()


@pytest.mark.parametrize(
    ("description", "expected_category"),
    [
        ("Combustível", Category.CARRO),
        ("Almoço em restaurante", Category.ALIMENTACAO),
        ("iFood", Category.IFOOD),
        ("Energia Elétrica", Category.CONTAS_FIXAS),
        ("Mercado", Category.ALIMENTACAO),
        ("Farmácia", Category.SAUDE),
        ("IPTU", Category.IMPOSTO),
        ("IPVA", Category.IMPOSTO),
        ("Netflix", Category.STREAMING),
        ("HBO Max", Category.STREAMING),
        ("AACD", Category.DOACAO),
        ("Internet Banda Larga", Category.CONTAS_FIXAS),
    ],
)
def test_classify_expense(classifier, description, expected_category):
    result = classifier.classify(description)

    assert result.category == expected_category
    assert result.description == description


def test_unknown_expense(classifier):
    with pytest.raises(ValueError):
        classifier.classify("Viagem para a praia")