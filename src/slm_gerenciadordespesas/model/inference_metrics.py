from dataclasses import dataclass


@dataclass
class InferenceMetrics:
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    eval_count: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int

    @property
    def total_duration_seconds(self) -> float:
        return self.total_duration_ns / 1_000_000_000

    @property
    def load_duration_seconds(self) -> float:
        return self.load_duration_ns / 1_000_000_000

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_eval_duration_ns == 0:
            return 0.0

        return (
            self.prompt_eval_count
            / (self.prompt_eval_duration_ns / 1_000_000_000)
        )

    @property
    def generation_tokens_per_second(self) -> float:
        if self.eval_duration_ns == 0:
            return 0.0

        return (
            self.eval_count
            / (self.eval_duration_ns / 1_000_000_000)
        )
