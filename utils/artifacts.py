import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(base_dir: str = "artifacts") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_agent_dir(run_dir: Path, agent_name: str) -> Path:
    agent_dir = run_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else str(content), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
