results = {
    "Rules": {
        "accuracy": 78.57,
        "avg_time": 0.000,
        "slm_rate": 0.0,
    },
    "Qwen 1.7B": {
        "accuracy": 89.29,
        "avg_time": 1.215,
        "slm_rate": 100.0,
    },
    "Qwen 4B": {
        "accuracy": 100.0,
        "avg_time": 2.538,
        "slm_rate": 100.0,
    },
    "Hybrid": {
        "accuracy": 100.0,
        "avg_time": 0.192,
        "slm_rate": 44.0,
    },
}

qwen_17 = results["Qwen 1.7B"]["avg_time"]
hybrid = results["Hybrid"]["avg_time"]

print("============================================================")
print("COMPARAÇÃO")
print("============================================================")

for name, result in results.items():
    print(
        f"{name:12} | "
        f"acurácia={result['accuracy']:6.2f}% | "
        f"tempo={result['avg_time']:6.3f}s | "
        f"SLM={result['slm_rate']:6.2f}%"
    )

print()
print(f"Speedup Hybrid vs Qwen 1.7B: {qwen_17 / hybrid:.2f}x")
print(f"Redução de chamadas SLM:      {100 - results['Hybrid']['slm_rate']:.2f}%")