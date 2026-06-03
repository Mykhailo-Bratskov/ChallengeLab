# Agentic AI for ML Challenge Execution

This project automates the full early phase of a machine learning competition workflow:

1. Read and structure challenge rules (URL or PDF).
2. Extract dataset metadata from zipped datasets (including EDF and tabular formats).
3. Run research to identify relevant methods.
4. Run data analysis for planner-ready numeric insights.
5. Build an actionable implementation plan.
6. Generate implementation code from that plan.

## Why this project is valuable

- Reduces manual setup time from hours to minutes.
- Enforces a structured workflow across research, data understanding, and planning.
- Produces planner-ready outputs with explicit assumptions and risks.
- Supports heavy/heterogeneous datasets by using agentic remote analysis when needed.
- Tracks token usage to improve cost visibility.

## Current architecture

- `app/main.py`: Orchestrates the end-to-end pipeline.
- `agents/challenge_agent.py`: Extracts contest rules from URL/PDF.
- `agents/metadata_extraction.py`: Inspects extracted dataset files and builds metadata.
- `agents/research_agent.py`: Produces method and feature recommendations from rules + metadata.
- `agents/data_analysis_agent.py`: Produces numeric EDA-style findings for planning (supports agentic remote mode).
- `agents/planner_agent.py`: Converts research + data context into an execution plan.
- `agents/code_agent.py`: Generates code from the actionable plan.
- `utils/prompt_input.py`: CLI input helpers.
- `utils/token_counter.py`: Token accounting helper.
- `space_for_code`: folder where the code_agent will compile ML pipeline

## Requirements

- Python `3.10+`
- API keys:
  - `GEMINI_API_KEY` (required)
  - `ANTHROPIC_API_KEY` (required for `planner_agent.py` in current setup)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Run

Run the pipeline from project root:

```bash
python -m app.main
```

You will be prompted for:

1. Contest rules source (URL or PDF path).
2. Dataset zip path.
3. Optional precomputed data-analysis JSON, or permission to run analysis agent.

## Data handling notes

- Metadata extraction is done locally from extracted files.
- Data analysis agent can run in a fully agentic remote mode to reduce local compute pressure.
- For very large datasets, keep `max_files` and `max_total_mb` constrained when calling the analysis agent.

## Known limitations

- Agent outputs depend on model quality and prompt constraints.
- Some file formats may require fallback behavior if parsing fails.
- Planner currently depends on Anthropic model configuration in code.
- Important: The current data_analysis_agent.py implementation is experimental and may not produce reliable or complete analysis results. It is therefore recommended to perform your own EDA/data analysis and provide it as a precomputed JSON input when running the pipeline.
