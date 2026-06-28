from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def slugify(value: str, max_len: int = 64) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9äöüß]+", "-", value)
    value = value.strip("-")
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return (value[:max_len].strip("-") or "podcast")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_checked(cmd: list[str], *, input_text: str | None = None, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{result.returncode}: {' '.join(cmd)}\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
    return result


def json_from_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")

