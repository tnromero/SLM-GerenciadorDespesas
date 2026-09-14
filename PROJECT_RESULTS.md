# Projeto 1 — SLM Gerenciador de Despesas

## Objetivo

Construir um classificador de despesas utilizando um Small Language Model
(SLM) local, comparando uma abordagem determinística com uma abordagem
baseada em SLM e uma arquitetura híbrida.

## Stack

- Python 3.13
- Ollama
- Qwen3 1.7B
- Qwen3 4B
- Pydantic
- pytest
- uv
- NVIDIA GTX 1650 4 GB
- WSL2

## Arquiteturas avaliadas

### Rule Based

Classificação baseada em regras explícitas.

Vantagens:

- latência praticamente zero
- comportamento determinístico
- sem consumo de GPU
- fácil auditoria

Limitação:

- baixa capacidade de generalização

### SLM

Classificação realizada pelo Qwen.

Vantagens:

- entende variações semânticas
- não exige uma regra para cada frase
- melhor capacidade de generalização

Limitações:

- maior latência
- consumo de GPU
- dependência do modelo
- possibilidade de respostas inválidas

### Hybrid

Primeiro tenta classificação determinística.

Quando nenhuma regra é encontrada, utiliza o SLM.

```text
Despesa
   |
   v
Rule Engine
   |
   +---- encontrada ----> Categoria
   |
   +---- não encontrada -> Qwen
                              |
                              v
                           Categoria
```

## Resultados

### Dataset original

| Estratégia | Acurácia |
| ---------- | -------: |
| Rule Based |   78,57% |
| Qwen 1.7B  |   89,29% |
| Qwen 4B    |     100% |
| Hybrid     |     100% |

O Few-Shot Prompt melhorou o Qwen 1.7B de 75,00% para 89,29%.

No Qwen 4B, a acurácia passou de 96,43% para 100%.

### Generalização

Dataset separado contendo descrições semanticamente diferentes das
utilizadas nas regras originais.

| Estratégia | Acurácia |
| ---------- | -------: |
| Qwen 1.7B  |     100% |
| Qwen 4B    |     100% |
| Hybrid     |     100% |

O Rule Based obteve 0% porque as frases foram deliberadamente escolhidas
fora das regras existentes.

### Hybrid — dataset misto

Resultado:

- Acurácia: 100%
- Acertos: 25/25
- Chamadas ao SLM: 11/25
- Taxa de utilização do SLM: 44%
- Classificações resolvidas por regras: 14/25
- Tempo médio observado: aproximadamente 0,19 s

Isso demonstra que o SLM não precisa processar todas as entradas.

## Conclusões

O modelo maior apresentou a melhor acurácia isoladamente, mas com custo
de inferência maior.

O Qwen 1.7B apresentou capacidade suficiente para o problema após o uso
de Few-Shot Prompting.

A arquitetura híbrida apresentou o melhor equilíbrio entre:

- acurácia
- latência
- uso de recursos
- previsibilidade

No dataset misto, 56% das classificações foram resolvidas sem utilizar
o SLM.

## Conceitos aprendidos

- classificação determinística
- Small Language Models
- Ollama
- inferência local
- Few-Shot Prompting
- Structured Output
- JSON Schema
- Pydantic Validation
- Retry
- métricas de inferência
- avaliação de modelos
- dataset de generalização
- arquitetura híbrida
- observabilidade da origem da classificação

## Arquitetura final

```text
                    +----------------+
                    |    Despesa     |
                    +-------+--------+
                            |
                            v
                    +---------------+
                    | Rule Based    |
                    +-------+-------+
                            |
                  +---------+---------+
                  |                   |
             encontrada          não encontrada
                  |                   |
                  v                   v
             +---------+        +----------+
             | Expense |        | Qwen 1.7B|
             +---------+        +----+-----+
                                      |
                                      v
                              Pydantic Validation
                                      |
                                      v
                                   Expense
```

## Próximas evoluções

O projeto pode evoluir para:

1. classificação baseada em histórico do usuário
1. armazenamento das despesas
1. RAG
1. explicação da classificação
1. feedback do usuário
1. aprendizado baseado em correções
1. avaliação contínua
1. comparação com modelos maiores
1. quantização
1. segundo projeto utilizando contexto e histórico

## Decisão

Para este problema, a arquitetura recomendada é:

**Rule Based + Qwen3 1.7B como fallback.**

O Qwen3 4B demonstrou maior capacidade, mas o ganho não justifica
necessariamente o aumento de latência para este caso de uso.
