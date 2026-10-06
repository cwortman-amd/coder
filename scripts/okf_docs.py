#!/usr/bin/env python3
"""Migrate, index, and validate this repository as an OKF v0.2 bundle."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESERVED = {"index.md", "log.md"}
LOCAL_FILE_PREFIX = "file:///home/amd/workspace/coder/"
INDEX_PATHS = (
    Path("index.md"),
    Path("docs/index.md"),
    Path("docs/results/index.md"),
    Path("_results/index.md"),
)
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
DATE_HEADING_RE = re.compile(r"^##\s+\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)


DOC_OVERRIDES: dict[str, dict[str, Any]] = {
    "README.md": {
        "type": "Repository Guide",
        "tags": ["repository", "inference", "rocm"],
    },
    "docs/README.md": {
        "type": "Documentation Catalog",
        "tags": ["documentation", "catalog"],
    },
    "docs/presentation/coder-tco.md": {
        "type": "Presentation",
        "tags": ["presentation", "tco", "coder", "on-premises"],
    },
    "docs/presentation/tail-latency.md": {
        "type": "Presentation",
        "tags": ["presentation", "agentx", "tail-latency", "pdd"],
    },
    "docs/TAIL-EVALUATION-PLAN.md": {
        "type": "Evaluation Plan",
        "tags": ["evaluation", "tail-latency", "agentx"],
    },
    "docs/TESTPLAN.md": {
        "type": "Evaluation Plan",
        "tags": ["evaluation", "pdd", "inference"],
    },
}


def markdown_paths() -> list[Path]:
    """Return tracked and not-ignored untracked Markdown paths."""
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*.md",
        ],
        cwd=ROOT,
        text=True,
    )
    paths = {Path(line) for line in output.splitlines() if line.strip()}
    paths.update(path for path in INDEX_PATHS if (ROOT / path).is_file())
    return sorted(paths)


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str]:
    """Return parsed frontmatter, body, and the original YAML source."""
    if not text.startswith("---\n"):
        return None, text, ""
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("opening YAML delimiter has no closing delimiter")
    # Early migration builds emitted an empty-mapping marker when no inferred
    # keys were missing. Treat it as no-op metadata so reruns repair cleanly.
    source = re.sub(r"^(?:\{\}\n)+", "", text[4:end])
    parsed = yaml.safe_load(source) or {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return parsed, text[end + 5 :], source


def clean_inline_markdown(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_~]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" #:|-")
    return value


def title_from_body(path: Path, body: str) -> str:
    match = HEADING_RE.search(body)
    if match:
        title = clean_inline_markdown(match.group(1))
        if title:
            return title
    name = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(word.upper() if word.lower() in {"gpu", "pdd", "tp", "ep"} else word.title() for word in name.split())


def description_from_body(title: str, body: str) -> str:
    in_fence = False
    paragraph: list[str] = []
    for raw in body.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            if paragraph:
                break
            continue
        if (
            stripped.startswith(("#", "|", "![", "<!--", "---", "===", "```"))
            or stripped == title
        ):
            continue
        cleaned = clean_inline_markdown(stripped.lstrip("> "))
        if not cleaned:
            continue
        paragraph.append(cleaned)
        if len(" ".join(paragraph)) >= 100:
            break
    description = " ".join(paragraph)
    if not description:
        description = f"Repository knowledge for {title}."
    if len(description) > 220:
        description = description[:217].rsplit(" ", 1)[0] + "..."
    if description[-1:] not in ".!?":
        description += "."
    return description


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value


def infer_type(path: Path) -> str:
    key = path.as_posix()
    if key in DOC_OVERRIDES:
        return str(DOC_OVERRIDES[key]["type"])
    upper = path.name.upper()
    parts = {part.lower() for part in path.parts}
    if "_results" in parts:
        if "checks" in parts:
            return "Validation Report"
        if "profiling" in parts:
            return "Profiling Report"
        if "quality" in parts or "validation" in parts:
            return "Quality Evaluation"
        if "pd_emulator" in parts:
            return "Simulation Report"
        if "upstream" in parts:
            return "Issue Catalog"
        return "Benchmark Report"
    if "presentation" in parts:
        return "Presentation"
    if "results" in parts:
        return "Benchmark Report"
    if "PLAN" in upper or upper == "TESTPLAN.MD":
        return "Evaluation Plan"
    if path.name.lower() == "readme.md":
        return "Documentation Catalog"
    if "OPS" in upper or "BENCH" in upper:
        return "Operations Guide"
    return "Technical Report"


def infer_tags(path: Path, doc_type: str) -> list[str]:
    key = path.as_posix()
    if key in DOC_OVERRIDES and "tags" in DOC_OVERRIDES[key]:
        return list(DOC_OVERRIDES[key]["tags"])
    tags: list[str] = []
    type_tag = slug(doc_type)
    if type_tag:
        tags.append(type_tag)
    for part in path.parts[:-1]:
        normalized = slug(part)
        if normalized and normalized not in {"docs", "results", "profiling"}:
            tags.append(normalized)
    stem = slug(path.stem)
    for token in stem.split("-"):
        if token and token not in {"report", "summary", "result", "md"}:
            tags.append(token)
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped[:8] or ["documentation"]


def metadata_for(path: Path, body: str) -> dict[str, Any]:
    doc_type = infer_type(path)
    title = title_from_body(path, body)
    return {
        "type": doc_type,
        "title": title,
        "description": description_from_body(title, body),
        "tags": infer_tags(path, doc_type),
        "status": "stable",
    }


def dump_metadata(metadata: dict[str, Any]) -> str:
    return yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip()


def normalize_local_links(text: str) -> str:
    return text.replace(f"]({LOCAL_FILE_PREFIX}", "](/")


def ensure_frontmatter(
    body: str,
    *,
    doc_type: str,
    title: str,
    description: str,
    tags: Iterable[str],
    status: str = "stable",
) -> str:
    """Wrap generated Markdown in project-standard OKF concept metadata."""
    metadata = {
        "type": doc_type,
        "title": title,
        "description": description,
        "tags": list(tags),
        "status": status,
    }
    return f"---\n{dump_metadata(metadata)}\n---\n\n{body.lstrip()}"


def migrate_file(path: Path) -> bool:
    absolute = ROOT / path
    original = absolute.read_text(encoding="utf-8")
    if path.name.lower() in RESERVED:
        normalized = normalize_local_links(original)
        if normalized != original:
            absolute.write_text(normalized, encoding="utf-8")
            return True
        return False
    frontmatter, body, source = split_frontmatter(original)
    body = normalize_local_links(body)
    inferred = metadata_for(path, body)
    if frontmatter is None:
        updated = f"---\n{dump_metadata(inferred)}\n---\n\n{body.lstrip()}"
    else:
        additions = {
            key: value for key, value in inferred.items() if key not in frontmatter
        }
        prefix = dump_metadata(additions) if additions else ""
        merged_source = f"{prefix}\n{source}" if prefix else source
        updated = f"---\n{merged_source.rstrip()}\n---\n{body}"
    if updated != original:
        absolute.write_text(updated, encoding="utf-8")
        return True
    return False


def read_concept(path: Path) -> dict[str, Any]:
    metadata, _, _ = split_frontmatter((ROOT / path).read_text(encoding="utf-8"))
    return metadata or {}


def index_entry(index_path: Path, concept_path: Path) -> str:
    metadata = read_concept(concept_path)
    if concept_path.name.lower() == "index.md" and not metadata:
        catalog_names = {
            "_results/index.md": ("Archived Results Catalog", "archived benchmark and profiling results"),
            "docs/index.md": ("Documentation Catalog", "project documentation"),
            "docs/results/index.md": ("Published Results Catalog", "published benchmark results"),
        }
        title, scope = catalog_names.get(
            concept_path.as_posix(),
            ("Bundle Catalog", "the bundle"),
        )
        description = f"Progressive-disclosure index for {scope}."
    else:
        title = metadata.get("title") or concept_path.stem
        description = metadata.get("description") or f"{title}."
    relative = concept_path.relative_to(index_path.parent).as_posix()
    return f"* [{title}]({relative}) - {description}"


def write_index(path: Path, sections: list[tuple[str, list[Path]]]) -> None:
    lines: list[str] = []
    if path == Path("index.md"):
        lines.extend(["---", 'okf_version: "0.2"', "---", ""])
    for heading, concepts in sections:
        if not concepts:
            continue
        lines.extend([f"# {heading}", ""])
        lines.extend(index_entry(path, concept) for concept in sorted(concepts))
        lines.append("")
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_indexes() -> None:
    concepts = [
        path
        for path in markdown_paths()
        if path.name.lower() not in RESERVED and (ROOT / path).is_file()
    ]
    root_concepts = [path for path in concepts if len(path.parts) == 1]
    docs_top = [
        path for path in concepts if path.parts[0] == "docs" and len(path.parts) == 2
    ]
    docs_results = [
        path
        for path in concepts
        if len(path.parts) >= 3 and path.parts[:2] == ("docs", "results")
    ]
    archived = [path for path in concepts if path.parts[0] == "_results"]
    write_index(
        Path("docs/results/index.md"),
        [("Published Benchmark Results", docs_results)],
    )
    grouped: dict[str, list[Path]] = {}
    for concept in archived:
        group = concept.parts[1] if len(concept.parts) > 2 else "Top Level"
        grouped.setdefault(group.replace("_", " ").title(), []).append(concept)
    write_index(Path("_results/index.md"), sorted(grouped.items()))
    write_index(
        Path("docs/index.md"),
        [
            ("Guides and Technical Reports", docs_top),
            ("Published Results", [Path("docs/results/index.md")]),
            ("Presentation", [
                Path("docs/presentation/coder-tco.md"),
                Path("docs/presentation/tail-latency.md"),
            ]),
        ],
    )
    write_index(
        Path("index.md"),
        [
            ("Repository", root_concepts),
            ("Documentation", [Path("docs/README.md")]),
            ("Archived Results", [Path("_results/index.md")]),
        ],
    )


def validate_link(source: Path, destination: str) -> str | None:
    destination = unquote(destination.strip("<>"))
    if destination.startswith(("#", "http://", "https://", "mailto:")):
        return None
    if destination.startswith("file://"):
        return f"{source}: non-portable file URI: {destination}"
    target_text = destination.split("#", 1)[0].split("?", 1)[0]
    if not target_text:
        return None
    if target_text.startswith("/"):
        target = ROOT / target_text.lstrip("/")
    else:
        target = ROOT / source.parent / target_text
    # Directory links often point at generated figure/result trees. OKF treats
    # them as graph hints and explicitly tolerates targets written later.
    if target_text.lower().endswith(".md"):
        try:
            target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            return f"{source}: concept link escapes bundle: {destination}"
        if not target.exists():
            return f"{source}: missing concept target: {destination}"
    return None


def validate() -> list[str]:
    errors: list[str] = []
    paths = markdown_paths()
    root_index = ROOT / "index.md"
    if not root_index.is_file():
        errors.append("missing bundle-root index.md")
    for path in paths:
        absolute = ROOT / path
        if not absolute.is_file():
            errors.append(f"{path}: file does not exist")
            continue
        text = absolute.read_text(encoding="utf-8")
        try:
            metadata, body, _ = split_frontmatter(text)
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{path}: invalid YAML frontmatter: {exc}")
            continue
        name = path.name.lower()
        if name == "index.md":
            if path == Path("index.md"):
                if not metadata or str(metadata.get("okf_version")) != "0.2":
                    errors.append("index.md: root index must declare okf_version 0.2")
            elif metadata is not None:
                errors.append(f"{path}: non-root index.md must not have frontmatter")
        elif name == "log.md":
            if metadata is not None:
                errors.append(f"{path}: log.md must not have frontmatter")
            if body and not DATE_HEADING_RE.search(body):
                errors.append(f"{path}: log.md needs ISO date headings")
        else:
            if metadata is None:
                errors.append(f"{path}: concept has no YAML frontmatter")
            else:
                for key in ("type", "title", "description", "tags", "status"):
                    if not metadata.get(key):
                        errors.append(f"{path}: missing project-required {key}")
                if metadata.get("status") not in {"draft", "stable", "deprecated"}:
                    errors.append(f"{path}: invalid status {metadata.get('status')!r}")
        for match in LINK_RE.finditer(body):
            issue = validate_link(path, match.group(1))
            if issue:
                errors.append(issue)
    return errors


def migrate() -> int:
    changed = 0
    for path in markdown_paths():
        if migrate_file(path):
            changed += 1
    print(f"migrated {changed} Markdown files")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("migrate", "index", "validate", "all"),
        default="validate",
        nargs="?",
    )
    args = parser.parse_args()
    if args.command in {"migrate", "all"}:
        migrate()
    if args.command in {"index", "all"}:
        generate_indexes()
    if args.command in {"validate", "all"}:
        errors = validate()
        if errors:
            print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
            print(f"OKF validation failed with {len(errors)} error(s)", file=sys.stderr)
            return 1
        print(f"OKF v0.2 validation passed for {len(markdown_paths())} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
