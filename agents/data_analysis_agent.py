import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY was not loaded. Check your .env file.")

client = genai.Client(api_key=api_key)

# Cheap model with tool use.
MODEL_ID = "gemini-2.5-flash-lite"

ANALYSIS_PROMPT = """
You are a Data Analysis Agent supporting an ML Planner Agent.
You must perform analysis yourself using code execution where useful.

Task:
1) Inspect attached dataset file(s), including EDF/tabular formats.
2) Compute useful numeric analysis for planning:
   - descriptive stats (means, std, quantiles)
   - missingness percentages
   - class/label distributions and imbalance (if target-like column exists)
   - outlier prevalence estimates
   - duplicate prevalence
   - file-level irregularities/inaccuracies or parse limitations
3) No plots. Numeric outputs only.

Return ONLY valid JSON with schema:
{
  "dataset_summary": "short text",
  "key_numeric_patterns": ["..."],
  "distribution_notes": ["..."],
  "data_quality_and_inaccuracies": ["..."],
  "target_and_imbalance_notes": ["..."],
  "planner_recommendations": ["..."],
  "validation_warnings": ["..."]
}

Rules:
- Quantify claims with percentages when possible.
- If only partial files/samples were analyzed, state that explicitly.
- If EDF content cannot be fully decoded in runtime, extract and report all metadata-level insights available.
""".strip()


def _find_data_files(dataset_access: str) -> list[Path]:
    p = Path(dataset_access).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p}")

    supported = {
        ".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson", ".edf", ".xlsx", ".xls"
    }

    if p.is_file():
        return [p] if p.suffix.lower() in supported else []

    return [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in supported]


def _pick_files_for_remote_analysis(files: list[Path], max_files: int, max_total_mb: int) -> list[Path]:
    if not files:
        return []

    # Prefer representative, larger files while staying under upload budget.
    ordered = sorted(files, key=lambda x: x.stat().st_size, reverse=True)
    selected: list[Path] = []
    total_bytes = 0
    cap = max_total_mb * 1024 * 1024

    for f in ordered:
        sz = f.stat().st_size
        if len(selected) >= max_files:
            break
        if total_bytes + sz > cap and selected:
            continue
        selected.append(f)
        total_bytes += sz

    return selected


def _upload_files(files: list[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    uploaded_parts: list[Any] = []
    manifest: list[dict[str, Any]] = []
    # Gemini file upload currently rejects some legacy formats (e.g., .xls MIME).
    uploadable_suffixes = {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson", ".edf", ".xlsx"}

    for f in files:
        if f.suffix.lower() not in uploadable_suffixes:
            manifest.append(
                {
                    "name": f.name,
                    "suffix": f.suffix.lower(),
                    "size_bytes": f.stat().st_size,
                    "upload_status": "skipped: unsupported upload MIME/type",
                }
            )
            continue
        try:
            up = client.files.upload(file=str(f))
            uploaded_parts.append(up)
            manifest.append(
                {
                    "name": f.name,
                    "suffix": f.suffix.lower(),
                    "size_bytes": f.stat().st_size,
                    "upload_status": "ok",
                }
            )
        except Exception as exc:
            manifest.append(
                {
                    "name": f.name,
                    "suffix": f.suffix.lower(),
                    "size_bytes": f.stat().st_size,
                    "upload_status": f"failed: {exc}",
                }
            )

    return uploaded_parts, manifest


def run_data_analysis_agent(
    dataset_access: str | Path,
    metadata: str | dict[str, Any] = "",
    challenge_rules: str | dict[str, Any] = "",
    max_files: int = 3,
    max_total_mb: int = 200,
):
    dataset_access = str(dataset_access)
    if isinstance(metadata, dict):
        metadata = json.dumps(metadata, ensure_ascii=True)
    if isinstance(challenge_rules, dict):
        challenge_rules = json.dumps(challenge_rules, ensure_ascii=True)

    files = _find_data_files(dataset_access)
    if not files:
        raise ValueError("No supported dataset files found for analysis.")

    selected = _pick_files_for_remote_analysis(files, max_files=max_files, max_total_mb=max_total_mb)
    uploaded_parts, upload_manifest = _upload_files(selected)

    if not uploaded_parts:
        raise RuntimeError("No files could be uploaded for remote agentic analysis.")

    context_blob = {
        "metadata_context": metadata[:12000] if metadata else "",
        "challenge_rules_context": challenge_rules[:12000] if challenge_rules else "",
        "upload_manifest": upload_manifest,
        "selection_policy": {
            "max_files": max_files,
            "max_total_mb": max_total_mb,
            "note": "Files analyzed remotely by model tools to reduce local compute load.",
        },
    }

    contents = uploaded_parts + [
        ANALYSIS_PROMPT,
        f"CONTEXT_JSON:\n{json.dumps(context_blob, ensure_ascii=True)}",
    ]

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution)],
            temperature=0.1,
        ),
    )

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    total_tokens = getattr(usage, "total_token_count", prompt_tokens + output_tokens) if usage else (prompt_tokens + output_tokens)

    print("--- Data Analysis Agent Token Usage ---")
    print(f"Prompt Tokens (Input): {prompt_tokens}")
    print(f"Candidate Tokens (Output): {output_tokens}")
    print(f"Total Tokens: {total_tokens}")

    return response.text, (prompt_tokens, output_tokens), {
        "files_selected": [str(f) for f in selected],
        "upload_manifest": upload_manifest,
    }


def load_data_analysis_from_json(eda_json_path: str | Path) -> str:
    """
    Read precomputed EDA JSON and return as compact string for planner input.
    """
    p = Path(eda_json_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"EDA JSON file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return json.dumps(data, ensure_ascii=True)


def get_data_analysis_for_planner(
    dataset_access: str | Path,
    metadata: str | dict[str, Any] = "",
    challenge_rules: str | dict[str, Any] = "",
    eda_json_path: str | Path | None = None,
    max_files: int = 3,
    max_total_mb: int = 200,
):
    """
    Convenience wrapper for your current main flow:
    - If `eda_json_path` is provided, load EDA from file.
    - Otherwise run fully agentic remote analysis.
    """
    if eda_json_path:
        eda_text = load_data_analysis_from_json(eda_json_path)
        return eda_text, (0, 0), {"source": "local_eda_json", "path": str(eda_json_path)}

    analysis_text, tokens, debug = run_data_analysis_agent(
        dataset_access=dataset_access,
        metadata=metadata,
        challenge_rules=challenge_rules,
        max_files=max_files,
        max_total_mb=max_total_mb,
    )
    return analysis_text, tokens, debug
