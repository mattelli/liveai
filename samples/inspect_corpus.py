#!/usr/bin/env python3
"""
Inspect a semantic-search corpus pickle produced by ``store_corpus``.

Works with both corpus formats:

- **v2** (current): ``{version, mode, text_model_path, image_model_path, records, embeddings}``
- **v1** (legacy): ``{version, model_path, passages, embeddings}``

Examples
--------
    python samples/inspect_corpus.py --list
    python samples/inspect_corpus.py --corpus autosar
    python samples/inspect_corpus.py --corpus autosar --sources
    python samples/inspect_corpus.py --corpus autosar --kind image --limit 5
    python samples/inspect_corpus.py --corpus autosar --source AUTOSAR --grep "readiness"
    python samples/inspect_corpus.py --corpus autosar --record 0
    python samples/inspect_corpus.py --corpus autosar --json

Note: ``pickle.load`` executes arbitrary code, so only inspect corpora you
generated locally. The inspector never prints the full embeddings matrix.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embedding_search import (  # noqa: E402
    CORPUS_FILE,
    DEFAULT_CORPORA_DIR,
    list_corpora,
    resolve_corpus_dir,
)

DEFAULT_CORPORA_PATH = REPO_ROOT / DEFAULT_CORPORA_DIR


# ------------------------------------------------------------------ loading


def load_payload(path: Path) -> dict:
    """Load a corpus pickle, preferring the project validator when available."""
    try:
        from src.embedding_search import load_corpus  # validates the format

        return load_corpus(path)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def as_records(payload: dict) -> list[dict]:
    """Normalize a v2 (records) or v1 (passages) payload to a records list."""
    if "records" in payload:
        return list(payload["records"])
    return [
        {"kind": "text", "source": "", "text": passage, "page": None}
        for passage in payload.get("passages", [])
    ]


# ------------------------------------------------------------------ summary


def summarize(payload: dict, records: list[dict], embeddings: np.ndarray) -> dict:
    kinds = Counter(record.get("kind", "text") for record in records)
    return {
        "version": payload.get("version", 1),
        "mode": payload.get("mode", "text"),
        "text_model_path": payload.get("text_model_path")
        or payload.get("model_path", ""),
        "image_model_path": payload.get("image_model_path", ""),
        "records": len(records),
        "text_records": kinds.get("text", 0),
        "image_records": kinds.get("image", 0),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_dtype": str(embeddings.dtype),
        "sources": len({record.get("source", "") for record in records}),
    }


def _checks(records: list[dict], embeddings: np.ndarray) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []
    rows_match = embeddings.ndim == 2 and embeddings.shape[0] == len(records)
    checks.append(("rows match records", bool(rows_match)))
    if embeddings.ndim == 2 and embeddings.shape[0]:
        norms = np.linalg.norm(embeddings, axis=1)
        checks.append(("L2-normalized", bool(np.allclose(norms, 1.0, atol=1e-4))))
    return checks


def _short(text, width: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def print_summary(path: Path, payload: dict, records: list[dict], embeddings: np.ndarray) -> None:
    info = summarize(payload, records, embeddings)
    print(f"Corpus:      {path}  ({path.stat().st_size:,} bytes)")
    print(f"Format:      v{info['version']}  mode={info['mode']}")
    print(f"text model:  {info['text_model_path'] or '-'}")
    print(f"image model: {info['image_model_path'] or '-'}")
    print(
        f"Records:     {info['records']}  "
        f"(text: {info['text_records']}, image: {info['image_records']})"
    )
    print(f"Embeddings:  {tuple(info['embeddings_shape'])} {info['embeddings_dtype']}")
    print(f"Sources:     {info['sources']}")
    checks = _checks(records, embeddings)
    print("Checks:      " + " | ".join(f"{name}: {'OK' if ok else 'FAIL'}" for name, ok in checks))


# ------------------------------------------------------------------ selection


def select_records(records: list[dict], args) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    for index, record in enumerate(records):
        if args.kind and record.get("kind", "text") != args.kind:
            continue
        if args.source and args.source.lower() not in record.get("source", "").lower():
            continue
        if args.grep and args.grep.lower() not in record.get("text", "").lower():
            continue
        selected.append((index, record))
    return selected


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a named local corpus.")
    parser.add_argument("--corpus", help="corpus name (see --list)")
    parser.add_argument("--list", action="store_true", dest="list_corpora", help="list available corpora")
    parser.add_argument(
        "--corpora-dir", type=Path, default=DEFAULT_CORPORA_PATH,
        help=f"corpora root (default: {DEFAULT_CORPORA_PATH})",
    )
    parser.add_argument("--sources", action="store_true", help="list source files with record counts")
    parser.add_argument("--kind", choices=("text", "image"), help="only show records of this kind")
    parser.add_argument("--source", help="only show records whose source contains this substring")
    parser.add_argument("--grep", help="only show records whose text contains this substring")
    parser.add_argument("--limit", type=int, default=10, help="max records to print (default: 10)")
    parser.add_argument("--record", type=int, metavar="N", help="print a single record by index (full text)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser


def _list_corpora(corpora_dir: Path) -> int:
    entries = list_corpora(corpora_dir)
    if not entries:
        print("No corpora available.")
        return 0
    print(f"Available corpora ({len(entries)}):")
    for e in entries:
        print(
            f"- {e.get('name')}: mode={e.get('mode')}, records={e.get('records')} "
            f"(text {e.get('text_records')} / image {e.get('image_records')}), "
            f"sources={e.get('sources')}, built={e.get('built_at')}"
        )
    return 0


def _print_record(index: int, record: dict, embeddings: np.ndarray) -> None:
    print(f"Record #{index}:")
    for key, value in record.items():
        print(f"  {key}: {value}")
    if embeddings.ndim == 2 and 0 <= index < embeddings.shape[0]:
        print(f"  embedding[:8]: {np.round(embeddings[index][:8], 5).tolist()}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    corpora_dir: Path = args.corpora_dir

    if args.list_corpora:
        return _list_corpora(corpora_dir)

    if not args.corpus:
        print("--corpus <name> is required (or use --list)", file=sys.stderr)
        return 1
    try:
        corpus_dir = resolve_corpus_dir(args.corpus, corpora_dir)
    except ValueError as exc:
        print(f"Invalid corpus name: {exc}", file=sys.stderr)
        return 1

    path: Path = corpus_dir / CORPUS_FILE
    if not path.exists():
        print(f"Corpus not found: {path}", file=sys.stderr)
        return 1

    payload = load_payload(path)
    records = as_records(payload)
    embeddings = np.asarray(payload.get("embeddings"))

    if args.sources:
        counts = Counter(record.get("source", "") for record in records)
        if args.json:
            print(json.dumps({**summarize(payload, records, embeddings),
                              "source_counts": counts.most_common()}, indent=2))
            return 0
        print_summary(path, payload, records, embeddings)
        print("\nSources:")
        for source, count in counts.most_common():
            print(f"  {count:>6}  {source}")
        return 0

    if args.record is not None:
        if not 0 <= args.record < len(records):
            print(f"Record index out of range: {args.record} (0..{len(records) - 1})", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(records[args.record], ensure_ascii=False, indent=2))
        else:
            print_summary(path, payload, records, embeddings)
            print()
            _print_record(args.record, records[args.record], embeddings)
        return 0

    selected = select_records(records, args)
    if args.json:
        print(json.dumps({
            **summarize(payload, records, embeddings),
            "shown": min(args.limit, len(selected)),
            "items": [{"index": index, **record} for index, record in selected[: args.limit]],
        }, ensure_ascii=False, indent=2))
        return 0

    print_summary(path, payload, records, embeddings)
    print(f"\nRecords ({len(selected)} match; showing {min(args.limit, len(selected))}):")
    for index, record in selected[: args.limit]:
        kind = record.get("kind", "text")
        page = record.get("page")
        page_info = f" p.{page}" if page else ""
        print(f"[{index}] ({kind}{page_info}) {record.get('source', '')}")
        print(f"    {_short(record.get('text', ''))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
