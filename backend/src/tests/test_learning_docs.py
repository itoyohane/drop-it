"""Keep portable learning docs and their offline example tied to source contracts."""

import ast
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

from backend.agent.checkpoints import CHECKPOINT_STEPS
from backend.config import Settings


ROOT = Path(__file__).resolve().parents[3]
LEARNING_DOCS = (
    ROOT / "docs/README.md",
    ROOT / "docs/tutorial-run-and-observe.md",
    ROOT / "docs/explanation-agent-architecture.md",
    ROOT / "docs/reference-api-tools-and-data.md",
    ROOT / "docs/howto-ai-application-interview.md",
)
REFERENCE = LEARNING_DOCS[3]


def _markdown_links(markdown):
    # Current docs use inline links, not reference-style links or nested targets.
    return re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", markdown)


def _local_target(document, target):
    parsed = urlsplit(target.strip("<>"))
    if parsed.scheme or parsed.netloc:
        return None
    return (document.parent / unquote(parsed.path)).resolve()


@pytest.mark.parametrize("document", (*LEARNING_DOCS, ROOT / "README.md"), ids=lambda p: p.name)
def test_learning_document_links_are_portable_and_exist(document):
    markdown = document.read_text(encoding="utf-8")
    for target in _markdown_links(markdown):
        resolved = _local_target(document, target)
        if resolved is None:
            continue
        assert resolved.is_relative_to(ROOT), f"Nonportable link in {document}: {target}"
        assert resolved.exists(), f"Broken link in {document}: {target}"


def test_learning_pages_are_reachable_and_cross_linked():
    root_links = {
        _local_target(ROOT / "README.md", target)
        for target in _markdown_links((ROOT / "README.md").read_text(encoding="utf-8"))
    }
    assert LEARNING_DOCS[0] in root_links
    index = LEARNING_DOCS[0]
    index_links = {
        _local_target(index, target)
        for target in _markdown_links(index.read_text(encoding="utf-8"))
    }
    assert set(LEARNING_DOCS[1:]) <= index_links
    for document in LEARNING_DOCS[1:]:
        links = {
            _local_target(document, target)
            for target in _markdown_links(document.read_text(encoding="utf-8"))
        }
        assert index in links, f"Missing learning-home link: {document}"


def test_reference_http_routes_match_fastapi_source():
    source = ast.parse((ROOT / "backend/src/main.py").read_text(encoding="utf-8"))
    expected = set()
    for node in ast.walk(source):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "app":
            continue
        if node.func.attr not in {"get", "post", "patch", "put", "delete"} or not node.args:
            continue
        path = node.args[0]
        if isinstance(path, ast.Constant) and isinstance(path.value, str):
            expected.add((node.func.attr.upper(), path.value))
    documented = set(re.findall(
        r"`(GET|POST|PATCH|PUT|DELETE) (/api/[^`\s]+)`",
        REFERENCE.read_text(encoding="utf-8"),
    ))
    assert expected, "No API routes found in source"
    assert documented == expected


def test_reference_settings_names_and_defaults_match_source():
    markdown = REFERENCE.read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `([A-Z][A-Z0-9_]+)` \| ([^|]+) \|$", markdown, re.MULTILINE))
    expected_names = {field.validation_alias for field in Settings.model_fields.values()}
    assert set(rows) == expected_names
    for field in Settings.model_fields.values():
        description = rows[field.validation_alias].strip()
        default = field.default
        if default is None:
            assert description.startswith(("无", "未指定"))
        elif isinstance(default, bool):
            assert description.startswith(str(default).lower())
        elif isinstance(default, (int, float)):
            number = re.match(r"\d+(?:\.\d+)?", description)
            assert number and float(number.group()) == default
        else:
            assert description.startswith(str(default)), field.validation_alias


def test_reference_checkpoint_steps_match_source():
    markdown = REFERENCE.read_text(encoding="utf-8")
    flow = re.search(r"```text\n(command_parsed.*?)\n```", markdown, re.DOTALL)
    assert flow
    names = tuple(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b|\bcompleted\b", flow.group(1)))
    assert names == CHECKPOINT_STEPS


def test_tutorial_offline_python_example_runs(capsys):
    markdown = LEARNING_DOCS[1].read_text(encoding="utf-8")
    snippets = re.findall(r"@'\n(.*?)\n'@ \| python -", markdown, re.DOTALL)
    assert len(snippets) == 1
    exec(compile(snippets[0], str(LEARNING_DOCS[1]), "exec"), {})
    assert capsys.readouterr().out.strip() == "['a', 'b', 'c'] 600 True"


def test_link_resolution_decodes_spaces_and_keeps_external_links_separate():
    document = LEARNING_DOCS[0]
    assert _local_target(document, "p0%20lists.md") == ROOT / "docs/p0 lists.md"
    assert _local_target(document, "https://example.com/docs") is None
    assert _local_target(document, "../../outside.md").is_relative_to(ROOT) is False
