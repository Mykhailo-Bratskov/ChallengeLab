# add metadata_usage for token count to see the prices
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field
from utils import artifacts

load_dotenv()

# you can choose another API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY was not loaded. Check your .env file.")

client = genai.Client(api_key=api_key)
MODEL_ID = "antigravity-preview-05-2026"


class ResearchFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title_or_topic: str
    source_type: Literal[
        "academic_paper",
        "benchmark",
        "official_documentation",
        "review_paper",
        "methodological_resource",
        "unknown",
    ]
    relevance_score: float = Field(ge=0, le=1)
    relevant_modalities: list[str] = Field(default_factory=list)
    method_summary: str
    required_data: list[str] = Field(default_factory=list)
    compatible_with_dataset: Literal["yes", "partial", "no", "unknown"]
    adaptation_needed: str
    implementation_difficulty: Literal["low", "medium", "high"]
    expected_value: Literal["low", "medium", "high"]
    risks_or_limitations: list[str] = Field(default_factory=list)


class FeatureRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_group: str
    scientific_rationale: str
    required_inputs: list[str] = Field(default_factory=list)
    preprocessing_steps: list[str] = Field(default_factory=list)
    example_features: list[str] = Field(default_factory=list)
    aggregation_level: Literal[
        "patient_level",
        "recording_level",
        "epoch_level",
        "channel_level",
        "site_level",
        "unknown",
    ]
    priority: Literal["low", "medium", "high"]
    leakage_risks: list[str] = Field(default_factory=list)


class ModelRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_family: str
    why_suitable: str
    input_feature_types: list[str] = Field(default_factory=list)
    validation_notes: str
    risks: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high"]


class ExperimentIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    objective: str
    feature_groups: list[str] = Field(default_factory=list)
    model_family: str
    validation_strategy: str
    metric: str
    expected_output: str
    priority: Literal["low", "medium", "high"]


class ResearchAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_interpretation: str
    research_keywords: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    relevant_findings: list[ResearchFinding] = Field(default_factory=list)
    feature_recommendations: list[FeatureRecommendation] = Field(default_factory=list)
    model_recommendations: list[ModelRecommendation] = Field(default_factory=list)
    ranked_experiment_ideas: list[ExperimentIdea] = Field(default_factory=list)
    methods_to_avoid: list[str] = Field(default_factory=list)
    open_questions_for_user_or_dataset_agent: list[str] = Field(default_factory=list)


# Expanded, highly specific prompt
RULES_PROMPT = """
You are a Principal Machine Learning Researcher. Your goal is to design a concrete, state-of-the-art (SOTA) approach for a data science challenge.

Step 1: Review the provided Contest Rules to understand the constraints (e.g., allowed libraries, data leakage rules, evaluation metrics).
Step 2: Review the provided Dataset Metadata (column names, folder structure).
Step 3: Use your web search tools to research recent academic literature, Kaggle write-ups,
and established resources to find the best algorithmic approaches for this specific type of data.

OUTPUT FORMAT:
Return valid JSON only, matching this schema:
- task_interpretation: string
- research_keywords: string[]
- sources: string[]
- relevant_findings: ResearchFinding[]
- feature_recommendations: FeatureRecommendation[]
- model_recommendations: ModelRecommendation[]
- ranked_experiment_ideas: ExperimentIdea[]
- methods_to_avoid: string[]
- open_questions_for_user_or_dataset_agent: string[]
""".strip()


def parse_research_output(raw_text: str) -> ResearchAgentOutput:
    """Validate and parse model JSON output into a strict Pydantic object."""
    return ResearchAgentOutput.model_validate_json(raw_text)


def _resolve_rules_content(challenge_agent_output: str) -> str:
    """
    Accept either:
    1) raw JSON string returned by challenge_agent, or
    2) a path to a JSON/text file containing that output.
    """
    if not isinstance(challenge_agent_output, str) or not challenge_agent_output.strip():
        raise ValueError("challenge_agent_output must be a non-empty string.")

    candidate = challenge_agent_output.strip()
    candidate_path = Path(candidate)
    if candidate_path.exists() and candidate_path.is_file():
        return candidate_path.read_text(encoding="utf-8")

    return candidate


def run_research(metadata: str, challenge_agent_output: str, artifacts_run_dir: str | None = None) -> str:
    rules_content = _resolve_rules_content(challenge_agent_output)
    run_dir = Path(artifacts_run_dir) if artifacts_run_dir else artifacts.create_run_dir()
    agent_dir = artifacts.get_agent_dir(run_dir, "research")
    artifacts.write_json(
        agent_dir / "input.json",
        {
            "metadata": metadata,
            "challenge_agent_output": challenge_agent_output,
            "resolved_rules_content": rules_content,
        },
    )

    # Pass well-labeled blocks in the text payload
    interaction_inline = client.interactions.create(
        agent=MODEL_ID,
        system_instruction=(
            "You are a Principal ML Researcher. "
            "You strictly adhere to constraints. "
            "You always return strict JSON matching the requested schema."
        ),
        input=[
            {"type": "text", "text": RULES_PROMPT},
            {"type": "text", "text": f"--- DATASET METADATA ---\n{metadata}"},
            {"type": "text", "text": f"--- CONTEST RULES SUMMARY ---\n{rules_content}"},
        ],
        environment="remote",
        tools=[
            {"type": "google_search"},
            {"type": "url_context"},
        ],
    )

    # The Interactions API utilizes .usage instead of .usage_metadata
    usage = getattr(interaction_inline, "usage", None)
    
    # The field names have also been updated
    prompt_tokens = getattr(usage, "total_input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "total_output_tokens", 0) if usage else 0
    
    # You can also access cached tokens if using context caching: 
    # cached_tokens = getattr(usage, "total_cached_tokens", 0)
    
    total_tokens = prompt_tokens + output_tokens
    
    print("--- Research Agent Token Usage ---")
    print(f"Prompt Tokens (Input): {prompt_tokens}")
    print(f"Candidate Tokens (Output): {output_tokens}")
    print(f"Total Tokens: {total_tokens}")

    artifacts.write_text(agent_dir / "output_raw.txt", interaction_inline.output_text or "")
    artifacts.write_json(
        agent_dir / "usage.json",
        {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "model": MODEL_ID,
            "environment": "remote",
        },
    )
    artifacts.write_text(
        agent_dir / "notes.md",
        (
            "# Research Agent Notes\n\n"
            f"- Model: `{MODEL_ID}`\n"
            f"- Prompt tokens: `{prompt_tokens}`\n"
            f"- Output tokens: `{output_tokens}`\n"
            "- Output files: `output_raw.txt`, `output_parsed.json` (if valid), `usage.json`\n"
            "- Purpose: produce research findings and structured experiment ideas from contest rules and dataset metadata.\n"
        ),
    )

    try:
        parsed_output = parse_research_output(interaction_inline.output_text or "")
        artifacts.write_json(agent_dir / "output_parsed.json", parsed_output.model_dump())
    except Exception as exc:
        artifacts.write_json(agent_dir / "parse_error.json", {"error": str(exc)})

    # Return plain model text to downstream planning.
    return interaction_inline.output_text, (prompt_tokens, output_tokens)
