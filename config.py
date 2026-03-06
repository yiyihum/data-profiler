from dataclasses import dataclass


@dataclass
class Config:
    vllm_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    data_path: str = ""
    test_data_path: str = ""  # Optional: test data for train/test field comparison
    description_path: str = ""
    # Micro analysis
    micro_sample_per_class: int = 10
    micro_batch_size: int = 5
    max_micro_rounds: int = 3
    adaptive_sample_size: int = 10
    # Hypothesis verification
    max_hypotheses: int = 15
    max_verification_retries: int = 3
    code_timeout: int = 60
    # Bridge iteration
    max_bridge_iterations: int = 2
    # Coverage matrix
    min_coverage_rate: float = 0.8
    # Output
    output_dir: str = "./output"
    firejail_enabled: bool = False
