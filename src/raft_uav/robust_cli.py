"""CLI wrapper exposing all robust Kalman update modes.

The historical base CLI built its parser with a literal robust-update choice list
that only exposed ``nis-inflate``.  The tracker/update stack already supports
Student-t and Huber covariance reweighting, so this wrapper extends the parser
choices while delegating execution to :mod:`raft_uav.cli`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from raft_uav import cli as _base_cli
from raft_uav.baselines.update_logic import ROBUST_UPDATE_MODES

_ROBUST_UPDATE_OPTION = "--robust-update"
_BASE_ROBUST_UPDATE_CHOICES = ["none", "nis-inflate"]
_EXPOSED_ROBUST_UPDATE_CHOICES = ["none", *ROBUST_UPDATE_MODES]
_ROBUST_UPDATE_HELP = (
    "robust update rule; nis-inflate inflates covariance past the NIS gate, "
    "student-t applies heavy-tailed covariance reweighting, and huber applies "
    "multivariate Huber covariance reweighting"
)


@contextmanager
def expose_heavy_tailed_robust_update_modes() -> Iterator[None]:
    """Temporarily extend the base CLI's ``--robust-update`` choices.

    The base dispatcher creates its parser inside ``main()``, so the least
    invasive public-surface patch is to adjust only the corresponding
    ``ArgumentParser.add_argument`` call while the parser is being constructed.
    """

    original_add_argument = argparse.ArgumentParser.add_argument

    def patched_add_argument(self: argparse.ArgumentParser, *args: Any, **kwargs: Any):
        if _ROBUST_UPDATE_OPTION in args and kwargs.get("choices") == _BASE_ROBUST_UPDATE_CHOICES:
            kwargs = dict(kwargs)
            kwargs["choices"] = _EXPOSED_ROBUST_UPDATE_CHOICES
            kwargs["help"] = _ROBUST_UPDATE_HELP
        return original_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = patched_add_argument
    try:
        yield
    finally:
        argparse.ArgumentParser.add_argument = original_add_argument


def main(argv: list[str] | None = None) -> int:
    """Run the base CLI with Student-t and Huber robust-update choices exposed."""

    with expose_heavy_tailed_robust_update_modes():
        return _base_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
