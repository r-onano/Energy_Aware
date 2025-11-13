from rich.console import Console
from rich.theme import Theme

_custom = Theme({"info": "cyan", "warn": "yellow", "err": "bold red", "ok": "green"})
_console = Console(theme=_custom)


def info(msg: str):
    _console.print(f"[info]{msg}[/info]")


def warn(msg: str):
    _console.print(f"[warn]{msg}[/warn]")


def err(msg: str):
    _console.print(f"[err]{msg}[/err]")


def ok(msg: str):
    _console.print(f"[ok]{msg}[/ok]")
