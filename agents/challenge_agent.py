# add metadata_usage for token count to see the prices 
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY was not loaded. Check your .env file.")

client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.5-flash"

RULES_PROMPT = """
Extract a machine-learning challenge brief from the provided source.

Return only valid JSON with this exact schema:
{
  "contest_name": "string or null",
  "task_type": "classification|regression|ranking|generation|forecasting|unknown",
  "target_description": "string or null",
  "primary_metric": "string or null",
  "metric_direction": "maximize|minimize|unknown",
  "data_description": ["..."],
  "expected_files": ["..."],
  "submission_requirements": ["..."],
  "validation_or_leakage_notes": ["..."],
  "eligibility": ["..."],
  "deadlines": ["..."],
  "judging_criteria": ["..."],
  "prizes": ["..."],
  "prohibited_or_disqualifications": ["..."],
  "other_important_rules": ["..."],
  "open_questions": ["..."],
  "source_type": "url|pdf"
}

Field guidance:
- "task_type" should describe the ML task, not the competition type.
- "target_description" should explain what the model must predict or generate.
- "primary_metric" should be the official evaluation metric if available.
- "metric_direction" should be "maximize" if higher is better, "minimize" if lower is better, otherwise "unknown".
- "data_description" should summarize available data files, columns, modalities, or dataset structure.
- "expected_files" should include files like train.csv, test.csv, sample_submission.csv, metadata files, images, EDF files, or other mentioned inputs.
- "submission_requirements" should include required prediction format, file format, columns, number of submissions, or code requirements.
- "validation_or_leakage_notes" should include any information relevant to local validation, hidden test sets, grouped data, time splits, external sites, or leakage risks.
- "open_questions" should include anything important that is missing or unclear from the source.

If a section is missing:
- use null for missing string fields,
- use [] for missing list fields,
- use "unknown" for unknown categorical fields.

Do not add markdown.
Do not wrap the JSON in code fences.
Return JSON only.
""".strip()


def is_url(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    parsed = urlparse(text.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _extract_from_url(url: str) -> str:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text=f"{RULES_PROMPT}\n\nUse this URL as the source: {url}"),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            tools=[types.Tool(url_context=types.UrlContext())],
            temperature=0.1,
        ),
    )
    
    print("--- Challenge Debrief Agent Token Usage ---")
    print(f"Prompt Tokens (Input): {response.usage_metadata.prompt_token_count}")
    print(f"Candidate Tokens (Output): {response.usage_metadata.candidates_token_count}")
    print(f"Total Tokens: {response.usage_metadata.total_token_count}")

    return response.text, (response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)


def _wait_for_operation(operation):
    while not operation.done:
        time.sleep(2)
        operation = client.operations.get(operation)
    return operation


def _extract_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported for local file input.")

    store = client.file_search_stores.create(
        config={
            "display_name": f"contest-rules-{pdf_path.stem}",
            "embedding_model": "models/gemini-embedding-2",
        }
    )

    operation = client.file_search_stores.upload_to_file_search_store(
        file=pdf_path,
        file_search_store_name=store.name,
        config={"display_name": pdf_path.name},
    )
    _wait_for_operation(operation)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=RULES_PROMPT,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[store.name])
                )
            ],
            temperature=0.1,
        ),
    )

    print("--- Challenge Debrief Agent Token Usage ---")
    print(f"Prompt Tokens (Input): {response.usage_metadata.prompt_token_count}")
    print(f"Candidate Tokens (Output): {response.usage_metadata.candidates_token_count}")
    print(f"Total Tokens: {response.usage_metadata.total_token_count}")

    return response.text, (response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
