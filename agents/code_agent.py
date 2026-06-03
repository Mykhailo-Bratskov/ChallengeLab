import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, List
from dotenv import load_dotenv
from google import genai
from utils import artifacts

load_dotenv()
# get the api key for the model
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY was not loaded.")

client = genai.Client(api_key=api_key)
MODEL_ID = "antigravity-preview-05-2026"

# Define where you want the final code to be saved locally
LOCAL_OUTPUT_DIR = Path("./space_for_code")
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_sample_code() -> Tuple[bool, Optional[Path]]:
    while True:
        confirm = input("Is a code sample folder provided? (yes/no):\n> ").strip().lower()
        if confirm == "yes":
            sample = input("Please provide the path to the local folder:\n> ").strip()
            folder_path = Path(sample)
            if folder_path.is_dir():
                return True, folder_path
            else:
                print("Error: Path is not a directory. Try again.")
        elif confirm == "no":
            return False, None
        else:
            print("Please answer yes or no.")

# --- THE MAGIC HELPER FUNCTION ---
def folder_to_remote_sources(local_folder: Path) -> List[dict]:
    """Reads a local directory and converts it to Antigravity inline sources."""
    sources = []
    # Loop through the folder (you can add filters to ignore .git or __pycache__ etc)
    for file_path in local_folder.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".py", ".md", ".txt", ".json"]:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Calculate the relative path so the agent sees the correct folder structure
                relative_path = file_path.relative_to(local_folder)
                
                # Append it as an inline source targeting the remote workspace
                sources.append({
                    "type": "inline",
                    "target": f"/workspace/sample-code/{relative_path}",
                    "content": content
                })
            except Exception:
                pass # Skip files that aren't plain text
    return sources

def execute_code(dataset_access: str, actionable_plan: str, artifacts_run_dir: str | None = None) -> str:
    run_dir = Path(artifacts_run_dir) if artifacts_run_dir else artifacts.create_run_dir()
    agent_dir = artifacts.get_agent_dir(run_dir, "code")
    artifacts.write_json(
        agent_dir / "input.json",
        {
            "dataset_access": dataset_access,
            "actionable_plan": actionable_plan,
        },
    )

    have_code, sample_folder_path = get_sample_code()
    
    # 1. Base input payload
    input_payload = [
        {"type": "text", "text": f"--- DATASET ---\n{dataset_access}"},
        {"type": "text", "text": f"--- Plan for Code Implementation ---\n{actionable_plan}"},
    ]

    # 2. Build the sources array for the remote environment
    remote_sources = [
        {
            "type": "inline",
            "target": ".agents/AGENTS.md", 
            "content": "Follow the plan provided. You have a code sample in /workspace/sample-code/. Read it before generating new code.",
        }
    ]

    # If the user provided a folder, convert it and attach it to the environment!
    if have_code and sample_folder_path: 
        code_sources = folder_to_remote_sources(sample_folder_path)
        remote_sources.extend(code_sources)
        
    environment_config = {
        "type": "remote",
        "sources": remote_sources
    }

    print("\n Sending to remote Antigravity Agent...")

    # 3. Your original call structure
    interaction = client.interactions.create(
        agent=MODEL_ID,
        system_instruction=(
            "You are a Principal ML Engineer. You strictly adhere to the plan provided. "
            "CRITICAL: Your final output must include one final working Python script enclosed in ```python code blocks. "
            "The script must be executable locally, include if __name__ == '__main__', and print final metrics/results."
        ),
        input=input_payload,
        environment=environment_config,
    )

    # 4. Extract the code and save it back to your local machine
    final_response = interaction.output_text
    artifacts.write_text(agent_dir / "output_raw.txt", final_response or "")
    python_blocks = re.findall(r'```python\s*(.*?)\s*```', final_response, re.DOTALL)
    
    if python_blocks:
        final_code = python_blocks[-1]
        local_file_path = LOCAL_OUTPUT_DIR / "final_pipeline.py"
        local_file_path.write_text(final_code, encoding="utf-8")
        artifacts.write_text(agent_dir / "final_pipeline.py", final_code)
        print(f"\n SUCCESS: Final code successfully extracted and saved locally to {local_file_path}")
        run_generated_code(local_file_path, dataset_access)
    else:
        print("\n WARNING: No clean Python block was returned by the agent.")
    
    print("--- Code Agent Token Usage ---")
    # The Interactions API utilizes .usage instead of .usage_metadata
    usage = getattr(interaction, "usage", None)
    
    # The field names have also been updated
    prompt_tokens = getattr(usage, "total_input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "total_output_tokens", 0) if usage else 0
    
    # You can also access cached tokens if using context caching: 
    # cached_tokens = getattr(usage, "total_cached_tokens", 0)
    
    total_tokens = prompt_tokens + output_tokens
    print(f"Prompt Tokens (Input): {prompt_tokens}")
    print(f"Candidate Tokens (Output): {output_tokens}")
    print(f"Total Tokens: {total_tokens}")
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
            "# Code Agent Notes\n\n"
            f"- Model: `{MODEL_ID}`\n"
            f"- Dataset path passed: `{dataset_access}`\n"
            f"- Prompt tokens: `{prompt_tokens}`\n"
            f"- Output tokens: `{output_tokens}`\n"
            "- Output files: `output_raw.txt`, `final_pipeline.py` (if extracted), `usage.json`\n"
            "- Purpose: generate executable pipeline code from the actionable plan and run it locally.\n"
        ),
    )

    return interaction, (prompt_tokens, output_tokens)


def run_generated_code(script_path: Path, dataset_access: str) -> None:
    """
    Execute generated pipeline locally and print stdout/stderr for quick validation.
    """
    dataset_path = Path(dataset_access).expanduser()
    env = os.environ.copy()
    env["DATASET_PATH"] = str(dataset_path)

    print("\n Running generated pipeline locally...")
    try:
        cmd = [sys.executable, str(script_path), str(dataset_path)]
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        print(f"Generated pipeline exit code: {result.returncode}")
        if result.stdout.strip():
            print("\n--- Generated Pipeline STDOUT ---")
            print(result.stdout[-8000:])
        if result.stderr.strip():
            print("\n--- Generated Pipeline STDERR ---")
            print(result.stderr[-8000:])
        if result.returncode != 0:
            print("\n WARNING: Generated code execution failed. Review STDERR above.")
    except subprocess.TimeoutExpired:
        print("\n WARNING: Generated code timed out after 900 seconds.")
    except Exception as exc:
        print(f"\n WARNING: Could not run generated pipeline locally: {exc}")
