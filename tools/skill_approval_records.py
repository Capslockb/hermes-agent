"""Pending approval records for guarded agent-created skills."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_APPROVAL_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
_SAFE_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class SkillApprovalError(ValueError):
    """Raised when a skill approval record cannot be used."""


def approval_records_dir(home: Path | None = None) -> Path:
    return (home or get_hermes_home()) / "state" / "skill-approvals"


def _approval_record_path(approval_id: str, home: Path | None = None) -> Path:
    approval_id = (approval_id or "").strip()
    if not _APPROVAL_ID_RE.match(approval_id):
        raise SkillApprovalError("Invalid approval id.")
    return approval_records_dir(home) / f"{approval_id}.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_pending_skill_approval(
    home: Path,
    skill_dir: Path,
    scan_result: Any,
    reason: str,
) -> str:
    """Persist a pending approval record and a snapshot of the scanned skill."""
    approval_dir = approval_records_dir(home)
    approval_dir.mkdir(parents=True, exist_ok=True)

    safe_fragment = re.sub(r"[^A-Za-z0-9_]", "_", skill_dir.name[:24]).strip("_")
    base_id = f"sk{int(time.time())}_{safe_fragment or 'skill'}"
    approval_id = base_id
    counter = 1
    while (approval_dir / f"{approval_id}.json").exists():
        counter += 1
        approval_id = f"{base_id}_{counter}"

    snapshot_dir = approval_dir / approval_id / "skill"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(skill_dir, snapshot_dir, symlinks=False)

    record = {
        "id": approval_id,
        "skill_dir": str(skill_dir),
        "pending_skill_dir": str(snapshot_dir),
        "skill_name": scan_result.skill_name,
        "verdict": scan_result.verdict,
        "reason": reason,
        "findings_count": len(scan_result.findings),
        "findings": [
            {
                "pattern_id": f.pattern_id,
                "severity": f.severity,
                "category": f.category,
                "file": f.file,
                "line": f.line,
                "description": f.description,
            }
            for f in scan_result.findings
        ],
        "created_at": time.time(),
        "status": "pending",
    }
    _atomic_write_json(approval_dir / f"{approval_id}.json", record)
    return approval_id


def load_skill_approval(approval_id: str, home: Path | None = None) -> dict[str, Any]:
    path = _approval_record_path(approval_id, home)
    if not path.exists():
        raise SkillApprovalError(f"No approval record found for {approval_id}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillApprovalError(f"Approval record {approval_id} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise SkillApprovalError(f"Approval record {approval_id} is malformed.")
    return data


def list_pending_skill_approvals(home: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    approval_dir = approval_records_dir(home)
    if not approval_dir.exists():
        return records
    for path in sorted(approval_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("status") == "pending":
            records.append(data)
    return records


def _source_dir_for_record(record: dict[str, Any]) -> Path:
    for key in ("pending_skill_dir", "skill_dir"):
        value = record.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_dir() and (path / "SKILL.md").is_file():
            return path
    raise SkillApprovalError("The pending skill files are missing.")


def _safe_skill_name(record: dict[str, Any]) -> str:
    name = str(record.get("skill_name") or Path(str(record.get("skill_dir", ""))).name).strip()
    if not _SAFE_SKILL_NAME_RE.match(name):
        raise SkillApprovalError(f"Unsafe skill name in approval record: {name!r}.")
    return name


def approve_skill_approval(
    approval_id: str,
    *,
    home: Path | None = None,
    skills_root: Path | None = None,
) -> dict[str, Any]:
    """Install the pending skill snapshot and mark the record approved."""
    record = load_skill_approval(approval_id, home)
    if record.get("status") != "pending":
        raise SkillApprovalError(
            f"Approval {approval_id} is already {record.get('status', 'resolved')}."
        )

    source_dir = _source_dir_for_record(record)
    skill_name = _safe_skill_name(record)
    root = skills_root or ((home or get_hermes_home()) / "skills")
    root.mkdir(parents=True, exist_ok=True)
    dest = (root / skill_name).resolve()
    root_resolved = root.resolve()
    try:
        dest.relative_to(root_resolved)
    except ValueError as exc:
        raise SkillApprovalError("Resolved install path escapes the skills directory.") from exc

    tmp_dest = root / f".{skill_name}.approval-tmp"
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)
    shutil.copytree(source_dir, tmp_dest, symlinks=False)
    if dest.exists():
        shutil.rmtree(dest)
    tmp_dest.replace(dest)

    record["status"] = "approved"
    record["approved_at"] = time.time()
    record["installed_path"] = str(dest)
    _atomic_write_json(_approval_record_path(approval_id, home), record)

    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass

    return record


def deny_skill_approval(approval_id: str, *, home: Path | None = None) -> dict[str, Any]:
    """Mark a pending skill approval as denied."""
    record = load_skill_approval(approval_id, home)
    if record.get("status") != "pending":
        raise SkillApprovalError(
            f"Approval {approval_id} is already {record.get('status', 'resolved')}."
        )
    record["status"] = "denied"
    record["denied_at"] = time.time()
    _atomic_write_json(_approval_record_path(approval_id, home), record)
    return record


def format_pending_skill_approvals(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No pending skill approvals."
    lines = ["Pending skill approvals:"]
    for record in records:
        approval_id = record.get("id", "?")
        skill_name = record.get("skill_name", "?")
        findings = record.get("findings_count", 0)
        reason = record.get("reason", "")
        lines.append(f"- {approval_id}: {skill_name} ({findings} findings) {reason}")
    lines.append("")
    lines.append("Approve with `/skill-approve <id>` or deny with `/skill-deny <id>`.")
    return "\n".join(lines)
