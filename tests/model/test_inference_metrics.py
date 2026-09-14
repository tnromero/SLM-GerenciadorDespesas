from slm_gerenciadordespesas.model.inference_metrics import InferenceMetrics

def test_inference_metrics():
    metrics = InferenceMetrics(
        total_duration_ns=1000000,
        load_duration_ns=200000,
        prompt_eval_count=5,
        eval_count=10,
        prompt_eval_duration_ns=300000,
        eval_duration_ns=400000,
    )
    assert metrics.total_duration_ns == 1000000
    assert metrics.load_duration_ns == 200000
    assert metrics.prompt_eval_count == 5
    assert metrics.eval_count == 10
    assert metrics.prompt_eval_duration_ns == 300000
    assert metrics.eval_duration_ns == 400000