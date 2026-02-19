from __future__ import annotations

from agents.coder_agent import CoderAgent


def test_coder_returns_only_llm_candidates(monkeypatch) -> None:
    def fake_generate_candidates_via_llm(**kwargs):
        _ = kwargs
        return [
            {
                "description": "Candidate A",
                "code": "def check(x):\n    return bool(x)\n",
            },
            {
                "description": "Candidate B",
                "code": "def check(x):\n    return str(x).startswith('h')\n",
            },
        ]

    monkeypatch.setattr("tools.llm_ops.generate_candidates_via_llm", fake_generate_candidates_via_llm)

    coder = CoderAgent(
        llm_enabled=True,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key_env="OPENAI_API_KEY",
    )

    candidates = coder.generate_candidates(
        samples=[{"data": "hello", "label": "0"}],
        context={"task_description": "Analyze toxicity"},
        n_candidates=5,
    )

    assert len(candidates) == 2
    assert all(c.source == "llm" for c in candidates)
    assert candidates[0].description == "Candidate A"


def test_coder_returns_empty_when_llm_disabled() -> None:
    coder = CoderAgent(llm_enabled=False)
    candidates = coder.generate_candidates(
        samples=[{"data": "hello", "label": "0"}],
        context={"task_description": "Analyze toxicity"},
        n_candidates=5,
    )
    assert candidates == []
