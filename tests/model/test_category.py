from slm_gerenciadordespesas.model.category import Category

def test_values_category():
    
    assert Category.CARRO.value == "Carro"
    assert Category.IMPOSTO.value == "Imposto"
    assert Category.DOACAO.value == "Doação"
    assert Category.STREAMING.value == "Streaming"
    assert Category.IFOOD.value == "iFood"
    assert Category.SAUDE.value == "Saúde"
    assert Category.CONTAS_FIXAS.value == "Contas fixas"