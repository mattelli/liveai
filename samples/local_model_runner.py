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
  action (including starting/stopping the local server(s)), get the output,
  repeat until 'q'/Ctrl+C, so the terminal stays open.

``--start`` launches the requested local server(s) before connecting: the LM
Studio API server via ``lms server start``, the Ollama server via a detached
``ollama serve`` (so it keeps running after this script exits); servers that
already answer are left alone. ``--stop`` stops them again: ``lms server
stop`` for LM Studio, by terminating the Ollama serve process for Ollama.
Passing both flags restarts the selected server(s).

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
    python samples/local_model_runner.py --start lmstudio --list  # start LM Studio's server, then list
    python samples/local_model_runner.py --start ollama           # start Ollama, then the menu loop
    python samples/local_model_runner.py --start both             # start both servers, then the menu loop
    python samples/local_model_runner.py --stop ollama            # stop the Ollama server, then the menu loop
    python samples/local_model_runner.py --stop both --start both # restart both servers
    bin\\local.bat                                                # interactive loop via the venv wrapper
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import cprint
from src.local_model_handler import (LMStudioLocalModelHandler, LocalModelHandler,
                                     LocalModelHandlerError, OllamaLocalModelHandler, human_size)

VERBOSE_INFO_KEYS = ("modelinfo", "license", "template", "modelfile")  # verbose blobs
BACKENDS = ("all", "auto", "lmstudio", "ollama")
SERVER_CMDS = {"lmstudio": ("lms", "server", "start"), "ollama": ("ollama", "serve")}


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


def _backend_handler(backend: str):
    """A fresh single-backend handler (for probing reachability)."""
    return {"lmstudio": LMStudioLocalModelHandler, "ollama": OllamaLocalModelHandler}[backend]()


def probe_backend(backend: str) -> bool:
    """True when this single backend's server answers a cheap API call."""
    try:
        handler = _backend_handler(backend)
    except ImportError:
        return False  # SDK missing: start/stop_server() report the pip hint once
    try:
        return handler.is_available()
    except LocalModelHandlerError:
        return False


def start_server(backend: str, wait_seconds: float = 20.0) -> bool:
    """Start one local model server; True when it answers afterwards.

    Servers that already answer are left untouched. ``ollama serve`` runs in
    the foreground forever, so it is spawned detached (it survives this
    script; when Ollama is already up elsewhere it exits by itself); ``lms
    server start`` returns once the API server is up.
    """
    cmd = SERVER_CMDS[backend]
    try:
        _backend_handler(backend)
    except ImportError as exc:  # SDK missing: talking to it would fail anyway
        cprint(f"  {backend}: {exc}", "red")
        return False
    if probe_backend(backend):
        cprint(f"  {backend}: already running.", "grey")
        return True
    cprint(f"Starting {backend} ({' '.join(cmd)}) ...", "yellow", bold=True)
    try:
        if backend == "ollama":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP \
                if os.name == "nt" else 0
            subprocess.Popen(cmd, creationflags=flags, start_new_session=os.name != "nt",
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                cprint(f"  exit {result.returncode}"
                       + (f": {message}" if message else ""), "red")
    except FileNotFoundError:
        cprint(f"  '{cmd[0]}' not found on PATH (install it or start the server manually).", "red")
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if probe_backend(backend):
            cprint(f"  {backend}: server is up.", "green")
            return True
        time.sleep(1.0)
    cprint(f"  {backend}: not reachable after {wait_seconds:.0f}s.", "red")
    return False


def _apply_servers(selection: str, action) -> bool:
    """Run a per-server action (start/stop) on one backend or 'both'.

    True when every targeted server reached its target state.
    """
    backends = ("lmstudio", "ollama") if selection == "both" else (selection,)
    ok = True
    for backend in backends:
        ok &= action(backend)
    return ok


def start_servers(selection: str) -> bool:
    """Start the requested local model server(s): one backend or 'both'."""
    return _apply_servers(selection, start_server)


def _kill_ollama() -> None:
    """Terminate the Ollama server process (there is no stop command).

    ``taskkill`` targets every ollama.exe process (e.g. also a CLI session
    open elsewhere); ``pkill -f`` matches the ``ollama serve`` command line.
    """
    if os.name == "nt":
        subprocess.run(("taskkill", "/IM", "ollama.exe", "/F"), capture_output=True, text=True)
    else:
        subprocess.run(("pkill", "-f", "ollama serve"), capture_output=True, text=True)


def stop_server(backend: str, wait_seconds: float = 10.0) -> bool:
    """Stop one local model server; True when it no longer answers.

    A server that already does not answer is reported as not running. LM
    Studio is stopped with ``lms server stop``; the Ollama serve process has
    no stop command, so it is terminated directly (a running Ollama desktop
    app stays in the tray but no longer serves the API).
    """
    try:
        _backend_handler(backend)
    except ImportError as exc:  # SDK missing: cannot tell whether it runs
        cprint(f"  {backend}: {exc}", "red")
        return False
    if not probe_backend(backend):
        cprint(f"  {backend}: not running.", "grey")
        return True
    label = "lms server stop" if backend == "lmstudio" else "terminate ollama serve"
    cprint(f"Stopping {backend} ({label}) ...", "yellow", bold=True)
    try:
        if backend == "lmstudio":
            result = subprocess.run(("lms", "server", "stop"), capture_output=True, text=True)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                cprint(f"  exit {result.returncode}"
                       + (f": {message}" if message else ""), "red")
        else:
            _kill_ollama()
    except FileNotFoundError:
        cprint(f"  stop tool not found on PATH (stop the {backend} server manually).", "red")
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not probe_backend(backend):
            cprint(f"  {backend}: server stopped.", "green")
            return True
        time.sleep(0.5)
    cprint(f"  {backend}: still reachable after {wait_seconds:.0f}s.", "red")
    return False


def stop_servers(selection: str) -> bool:
    """Stop the requested local model server(s): one backend or 'both'."""
    return _apply_servers(selection, stop_server)


SERVER_ACTIONS = {"start": start_servers, "stop": stop_servers}


def prompt_servers(action: str) -> bool:
    """Ask which local server(s) to start/stop ('both' on empty input).

    ``action`` is 'start' or 'stop'; True when the selection was valid (even
    if a server then failed to start/stop); used by the interactive menu.
    """
    selection = input("Server [lmstudio/ollama/both] (Enter=both): ").strip().lower()
    if selection not in ("", "both", "all", "lmstudio", "ollama"):
        cprint(f"Unknown server {selection!r} (use lmstudio, ollama or both)", "red")
        return False
    SERVER_ACTIONS[action]("both" if selection in ("", "both", "all") else selection)
    return True


def interactive_loop(backend: str = "all", host: str | None = None) -> None:
    """Menu loop: pick an action, get the output, repeat until quit.

    The handler is (re)created whenever the backend changes; 'q', Ctrl+C or
    EOF leave the loop; 's'/'x' start/stop one or both local servers and
    force a rebind so the handler picks up (or drops) those backends.
    Server/backend errors are printed and the loop keeps running so the
    terminal stays open.
    """
    handler: LocalModelHandler | None = None
    bound: tuple[str, str | None] | None = None
    while True:
        if bound != (backend, host):
            try:
                handler = LocalModelHandler(backend, host=host)
            except (LocalModelHandlerError, ValueError) as exc:
                cprint(f"Error: {exc}", "red")
                choice = input("Backend [all/auto/lmstudio/ollama, Enter=retry, "
                               "s=start server(s), q=quit]: ").strip()
                if choice.lower() in ("q", "quit", "exit"):
                    return
                if choice.lower() == "s":
                    prompt_servers("start")  # the bind is retried on the next iteration
                elif choice:
                    backend = choice
                continue
            bound = (backend, host)
            cprint(f"\nBackend(s): {', '.join(handler.backends)}", "green", bold=True)

        print(
            "\n  1) Available models        2) Loaded models        3) Overview\n"
            "  4) Load a model            5) Model info           6) Unload a model\n"
            "  s) Start server(s)         x) Stop server(s)\n"
            f"  b) Switch backend (now: {backend})                  q) Quit"
        )
        try:
            choice = input("Choose [1-6, b, s, x, q, Enter=overview]: ").strip()
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
            elif choice.lower() == "s":
                if prompt_servers("start"):
                    bound = None  # rebind so newly started backends join the handler
            elif choice.lower() == "x":
                if prompt_servers("stop"):
                    bound = None  # rebind so stopped backends drop out of the handler
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
    parser.add_argument("-s", "--start", metavar="SERVER", choices=("lmstudio", "ollama", "both"),
                        help="start the LM Studio API server ('lms server start') and/or the "
                             "Ollama server ('ollama serve') before connecting; 'both' starts both")
    parser.add_argument("-x", "--stop", metavar="SERVER", choices=("lmstudio", "ollama", "both"),
                        help="stop the LM Studio API server ('lms server stop') and/or the "
                             "Ollama server process before connecting; 'both' stops both; "
                             "combine with --start to restart")
    parser.add_argument("--list", action="store_true", help="list available (downloaded/pulled) models")
    parser.add_argument("--loaded", action="store_true", help="show models currently loaded in memory")
    parser.add_argument("--load", metavar="MODEL", help="load this model and print its info")
    parser.add_argument("--unload", metavar="MODEL", help="unload this model")
    parser.add_argument("--info", metavar="MODEL", help="print info about this model")
    parser.add_argument("--host", metavar="HOST", help="explicit server host (e.g. localhost:1234 for LM Studio)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="menu loop after the one-shot actions (default when no action flag is given)")
    args = parser.parse_args()

    stopped_ok = True
    if args.stop:  # --stop before --start turns the pair into a restart
        stopped_ok = stop_servers(args.stop)
    if args.start:
        start_servers(args.start)
    any_action = args.list or args.loaded or args.load or args.info or args.unload
    if args.stop and not args.start and not any_action and not args.interactive:
        return 0 if stopped_ok else 1  # stop-only run: nothing else to do

    kwargs = {"host": args.host} if args.host else {}
    interactive = args.interactive or not any_action
    try:
        handler = LocalModelHandler(args.backend, **kwargs)
    except (LocalModelHandlerError, ValueError) as exc:
        cprint(f"Error: {exc}", "red")
        if not interactive:
            return 1
        interactive_loop(args.backend, args.host)  # its error branch offers 's'
        return 0

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
        if not interactive:
            return 1
        interactive_loop(args.backend, args.host)  # the menu offers 's' to start server(s)
        return 0

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