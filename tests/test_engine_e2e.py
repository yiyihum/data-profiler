from __future__ import annotations

from pathlib import Path

from agents.coder_agent import CoderAgent
from agents.ingestion_agent import IngestionAgent
from agents.sampler_agent import SamplerAgent
from core.compiler import ProfileCompiler
from core.engine import CDTEngine, EngineConfig
from core.memory import MetadataStore
from core.sandbox import LocalRestrictedRunner


def test_engine_end_to_end(tmp_path: Path, monkeypatch) -> None:
    def fake_generate_candidates_via_llm(**kwargs):
        _ = kwargs
        return [
            {
                "description": "LLM mock: offensive language detector",
                "code": (
                    "def check(x):\n"
                    "    text = '' if x is None else str(x).lower()\n"
                    "    return 'idiot' in text or 'fuck' in text\n"
                ),
            }
        ]

    monkeypatch.setattr("tools.llm_ops.generate_candidates_via_llm", fake_generate_candidates_via_llm)

    fixture = Path("scripts/fixtures/tiny_train.csv").resolve()
    sandbox = LocalRestrictedRunner(tmp_dir=tmp_path / "tmp", timeout_s=3, project_root=Path.cwd())

    ingestion = IngestionAgent(sandbox_runner=sandbox)
    ingested = ingestion.run(fixture, tmp_path)

    engine = CDTEngine(
        sampler=SamplerAgent(),
        coder=CoderAgent(llm_enabled=True, llm_api_key_env="OPENAI_API_KEY"),
        sandbox=sandbox,
        memory=MetadataStore(),
    )

    result = engine.run(
        parquet_path=str(ingested.parquet_path),
        config=EngineConfig(max_depth=2, min_samples_split=2, samples_per_node=3, candidates_per_node=3),
        metadata=ingested.metadata,
    )

    compiler = ProfileCompiler()
    output_path = tmp_path / "profile.json"
    profile = compiler.compile(result, output_path)

    assert output_path.exists()
    assert "directives" in profile
    assert len(profile["directives"]) >= 1
    assert profile["meta"]["description_source"] in {"file", "default"}
    assert isinstance(profile["meta"].get("description_excerpt"), str)
