"""Deprecated shim. Use: python main.py --chat"""
from main import main

if __name__ == "__main__":
    raise SystemExit(main(["--chat"]))
