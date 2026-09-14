from slm_gerenciadordespesas.controller.ollama_classifier import OllamaClassifier
from slm_gerenciadordespesas.config import OllamaConfig

classifier = OllamaClassifier(config=OllamaConfig(model="qwen3:4b"))

result = classifier.classify(
    "Assinei um serviço de streaming"
)

metrics = classifier.last_metrics

print(f"Resultado: {result.category.value}")
print()
print(f"Tempo total:       {metrics.total_duration_seconds:.3f}s")
print(f"Tempo carregamento: {metrics.load_duration_seconds:.3f}s")
print(f"Tokens prompt:     {metrics.prompt_eval_count}")
print(f"Tokens resposta:   {metrics.eval_count}")
print(f"Prompt tok/s:      {metrics.prompt_tokens_per_second:.2f}")
print(f"Geração tok/s:     {metrics.generation_tokens_per_second:.2f}")
