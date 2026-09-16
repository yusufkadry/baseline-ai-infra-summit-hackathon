#!/usr/bin/env python3
"""Compatibility entry point for standalone grid-tile WATCH evaluation."""

from __future__ import annotations

import sys

from tripwire import cli


if __name__ == "__main__":
    raise SystemExit(cli(["watch", *sys.argv[1:]]))
