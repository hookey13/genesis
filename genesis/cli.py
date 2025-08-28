#!/usr/bin/env python
"""
Main CLI entry point for Project GENESIS.

This module provides the command-line interface for all
Genesis operations including health checks and trading.
"""

import click

from genesis.cli.doctor import doctor


@click.group()
@click.version_option(version="1.0.0", prog_name="Genesis Trading System")
def cli():
    """Genesis Trading System - Tier-based algorithmic trading platform."""
    pass


# Register commands
cli.add_command(doctor)


if __name__ == "__main__":
    cli()
