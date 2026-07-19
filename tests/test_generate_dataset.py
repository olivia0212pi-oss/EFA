from types import SimpleNamespace

from common.io import write_jsonl
from generation.generate_dataset import _build_record, _truncated_items


def test_truncated_items_skips_records_with_a_final_answer(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    write_jsonl(
        path,
        [
            {
                "sample_id": "0",
                "question": "1+1?",
                "ground_truth": "2",
                "reasoning_text": "Thus \\boxed{2}.",
                "total_tokens": 10,
            },
            {
                "sample_id": "1",
                "question": "hard one",
                "ground_truth": "42",
                "reasoning_text": "still thinking, no answer yet",
                "total_tokens": 4096,
            },
        ],
    )
    items = _truncated_items(str(path))
    assert len(items) == 1
    assert items[0]["id"] == "1"
    assert items[0]["problem"] == "hard one"
    assert items[0]["answer"] == "42"
    assert items[0]["_previous_total_tokens"] == 4096


def test_build_record_records_retry_provenance() -> None:
    item = {"id": "1", "problem": "hard one", "answer": "42"}
    output = SimpleNamespace(text="\\boxed{42}", token_ids=[1, 2], logprobs=None)
    config = {
        "model": {"generation": "fake-model"},
        "data": {"dataset": "d", "split": "test"},
        "seed": 0,
        "generation": {"max_tokens": 8192},
    }
    retry_source = {"source": "orig.jsonl", "previous_total_tokens": 4096}
    record = _build_record(item, 0, output, 1.0, config, retry_source=retry_source)
    assert record["metadata"]["retried_from"] == {
        "source": "orig.jsonl",
        "previous_total_tokens": 4096,
    }
    assert record["final_answer"] == "42"
