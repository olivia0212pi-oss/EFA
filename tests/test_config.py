from common.config import load_config


def test_child_config_deep_merges_base() -> None:
    config = load_config("configs/smoke.yaml")
    assert config["data"]["num_samples"] == 5
    assert config["generation"]["max_tokens"] == 1024
    assert config["generation"]["temperature"] == 0.6
    assert config["model"]["generation"].endswith("Qwen-7B")

