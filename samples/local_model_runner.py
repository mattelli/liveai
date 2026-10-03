#!/usr/bin/env python3
"""
Demo of the local model-server handler (src/local_model_handler.py).

Lists available + loaded models across the local model servers, optionally
loads one and prints its info, then unloads it again. Works against LM Studio
(port 1234) and Ollama (port 11434); the default 'all' backend aggregates
every reachable server (each entry is tagged with its backend).

Modes:

* with action flags: one-shot, only the requested actions run;
* without action flags (or with ``-i``): an interactive menu loop - pick an
  action, get the output, repeat until 'q'/Ctrl+C, so the terminal stays open.

Model names may be qualified ("ollama:llama3.2:3b") when the same name exists
on several servers.

Examples
--------
    python samples/local_model_runner.py                          # interactive menu loop
    python samples/local_model_runner.py -i --loaded              # one-shot, then the menu loop
    python samples/local_model_runner.py --loaded                 # loaded on every server (one-shot)
    python samples/local_model_runner.py ollama --loaded          # only Ollama
    python samples/local_model_runner.py --list                   # only available models
    python samples/local_model_runner.py lmstudio --host localhost:1234
    python samples/local_model_runner.py --load "qwen3.5:0.8b" --loaded
    python samples/local_model_runner.py --info ollama:llama3.2:3b
    python samples/local_model_runner.py --unload llama3.2:3b
    bin\\local.bat                                                # interactive loop via the venv wrapper
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import cprint
from src.local_model_handler import LocalModelHandler, LocalModelHandlerError, human_size

VERBOSE_INFO_KEYS = ("modelinfo", "license", "template", "modelfile")  # verbose blobs
BACKENDS = ("all", "auto", "lmstudio", "ollama")


def print_models(title: str, models: list[dict]) -> None:
    cprint(f"\n{title} ({len(models)}):", "yellow", bold=True)
    if not models:
        cprint("  (none)", "grey")
        return
    for m in models:
        key = m.get("model_key") or m.get("model") or "?"
        size = human_size(m.get("size_bytes") or m.get("size") or 0)
        parts = [f"- {key} [{size}]"]
        for field in ("display_name", "architecture", "params_string",
                      "parameter_size", "quantization_level", "family"):
            if m.get(field):
                parts.append(f"{field}={m[field]}")
        if m.get("size_vram"):
            parts.append(f"vram={human_size(m['size_vram'])}")
        if m.get("context_length"):
            parts.append(f"ctx={m['context_length']}")
        if m.get("identifier"):
            parts.append(f"instance={m['identifier']}")
        if m.get("loaded"):
            parts.append("LOADED")
        cprint("  " + " ".join(parts), "cyan")


def print_grouped(handler: LocalModelHandler, title: str, entries: list[dict]) -> None:
    """One section per bound backend when several are bound."""
    if len(handler.backends) < 2:
        print_models(title, entries)
        return
    for backend in handler.backends:
        print_models(f"{title} [{backend}]",
                     [m for m in entries if m.get("backend") == backend])


def print_info(model: str, info: dict) -> None:
    cprint(f"\nInfo for '{model}':", "yellow", bold=True)
    for key, value in info.items():
        if key in VERBOSE_INFO_KEYS:
            continue  # verbose blobs
        cprint(f"  {key} = {value}", "cyan")


def do_load(handler: LocalModelHandler, model: str) -> None:
    cprint(f"\nLoading '{model}' ...", "yellow", bold=True)
    info = handler.load(model)
    cprint("  loaded: " + ", ".join(f"{k}={v}" for k, v in info.items() if v is not None), "cyan")
    print_grouped(handler, "Currently loaded models", handler.list_loaded())


def do_unload(handler: LocalModelHandler, model: str) -> None:
    cprint(f"\nUnloading '{model}': "
           f"{'done' if handler.unload(model) else 'not loaded'}", "yellow", bold=True)
    print_grouped(handler, "Currently loaded models", handler.list_loaded())


def prompt_model(label: str) -> str:
    """Ask for a model name; empty input skips the action."""
    model = input(f"{label} (e.g. llama3.2:3b, lmstudio:qwen/qwen3.5-4b, Enter=skip): ").strip()
    if not model:
        cprint("(skipped)", "grey")
    return model


def interactive_loop(backend: str = "all", host: str | None = None) -> None:
    """Menu loop: pick an action, get the output, repeat until quit.

    The handler is (re)created whenever the backend changes; 'q', Ctrl+C or
    EOF leave the loop. Server/backend errors are printed and the loop keeps
    running so the terminal stays open.
    """
    handler: LocalModelHandler | None = None
    bound: tuple[str, str | None] | None = None
    while True:
        if bound != (backend, host):
            try:
                handler = LocalModelHandler(backend, host=host)
            except (LocalModelHandlerError, ValueError) as exc:
                cprint(f"Error: {exc}", "red")
                choice = input("Backend [all/auto/lmstudio/ollama, Enter=retry, q=quit]: ").strip()
                if choice.lower() in ("q", "quit", "exit"):
                    return
                if choice:
                    backend = choice
                continue
            bound = (backend, host)
            cprint(f"\nBackend(s): {', '.join(handler.backends)}", "green", bold=True)

        print(
            "\n  1) Available models        2) Loaded models        3) Overview\n"
            "  4) Load a model            5) Model info           6) Unload a model\n"
            f"  b) Switch backend (now: {backend})                  q) Quit"
        )
        try:
            choice = input("Choose [1-6, b, q, Enter=overview]: ").strip()
        except EOFError:
            cprint("\nBye!", "grey")
            return
        except KeyboardInterrupt:
            cprint("", "grey")
            return
        if choice.lower() in ("q", "quit", "exit"):
            return

        try:
            if choice == "1":
                print_grouped(handler, "Available models", handler.list_models())
            elif choice == "2":
                print_grouped(handler, "Currently loaded models", handler.list_loaded())
            elif choice in ("", "3"):
                print_grouped(handler, "Available models", handler.list_models())
                print_grouped(handler, "Currently loaded models", handler.list_loaded())
            elif choice == "4":
                if model := prompt_model("Model to load"):
                    do_load(handler, model)
            elif choice == "5":
                if model := prompt_model("Model for info"):
                    print_info(model, handler.model_info(model))
            elif choice == "6":
                if model := prompt_model("Model to unload"):
                    do_unload(handler, model)
            elif choice.lower() == "b":
                new_backend = input(
                    f"Backend [{'/'.join(BACKENDS)}] (Enter keeps {backend}): ").strip().lower()
                if new_backend and new_backend != backend:
                    if new_backend not in BACKENDS:
                        cprint(f"Unknown backend {new_backend!r} (kept {backend})", "red")
                    else:
                        backend, bound = new_backend, None  # rebind on the next iteration
            else:
                cprint(f"Unknown choice {choice!r}", "grey")
        except EOFError:
            cprint("\nBye!", "grey")
            return
        except KeyboardInterrupt:
            cprint("(cancelled)", "grey")
        except LocalModelHandlerError as exc:
            cprint(f"Error: {exc}", "red")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backend", nargs="?", default="all", choices=list(BACKENDS),
                        help="'all' aggregates every reachable server (default), 'auto' uses the "
                             "first reachable one, or pin a single backend")
    parser.add_argument("--list", action="store_true", help="list available (downloaded/pulled) models")
    parser.add_argument("--loaded", action="store_true", help="show models currently loaded in memory")
    parser.add_argument("--load", metavar="MODEL", help="load this model and print its info")
    parser.add_argument("--unload", metavar="MODEL", help="unload this model")
    parser.add_argument("--info", metavar="MODEL", help="print info about this model")
    parser.add_argument("--host", metavar="HOST", help="explicit server host (e.g. localhost:1234 for LM Studio)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="menu loop after the one-shot actions (default when no action flag is given)")
    args = parser.parse_args()

    kwargs = {"host": args.host} if args.host else {}
    try:
        handler = LocalModelHandler(args.backend, **kwargs)
    except (LocalModelHandlerError, ValueError) as exc:
        cprint(f"Error: {exc}", "red")
        return 1

    # The server(s) must answer before any action (e.g. LM Studio app running
    # but its API server not started: "Local API host port is not yet resolved").
    hints = {
        "lmstudio": "start the LM Studio API server (Developer tab -> Start Server,"
                    " or run: lms server start)",
        "ollama": "start the Ollama server (run: ollama serve, or launch the Ollama app)",
    }
    if not handler.is_available():
        hint = hints.get(handler.backend, "start LM Studio (Developer tab -> Start Server,"
                                          " or: lms server start) and/or Ollama (ollama serve)")
        cprint(f"Error: no reachable model server.\nHint: {hint}", "red")
        return 1

    any_action = args.list or args.loaded or args.load or args.info or args.unload
    interactive = args.interactive or not any_action
    if not interactive:  # the loop prints its own (re)bind banner
        cprint(f"\nBackend(s): {', '.join(handler.backends)}", "green", bold=True)
    if any_action:  # one-shot actions (also run before the loop with -i)
        try:
            if args.list:
                print_grouped(handler, "Available models", handler.list_models())
            if args.loaded:
                print_grouped(handler, "Currently loaded models", handler.list_loaded())
            if args.load:
                do_load(handler, args.load)
            if args.info:
                print_info(args.info, handler.model_info(args.info))
            if args.unload:
                do_unload(handler, args.unload)
        except LocalModelHandlerError as exc:
            cprint(f"Error: {exc}", "red")
            if not args.interactive:
                return 1
    if interactive:
        interactive_loop(args.backend, args.host)

    return 0


if __name__ == "__main__":
    sys.exit(main())
