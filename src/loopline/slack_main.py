"""Entry point for the Slack privacy proxy.

Mirrors main.py but wires together the Slack client, the Slack privacy filter,
the Slack MCP server and the shared menu bar. Slack authenticates with a static
Bot User OAuth Token (``xoxb-...``) read from config, so there is no
``--oauth-setup`` command.

Threading model (identical to the Gmail proxy):
  - The MCP server runs the stdio transport inside its own asyncio event loop on
    a daemon background thread.
  - The rumps menu bar app runs on the main thread.
  - The shared ReviewQueue bridges the two via loop.call_soon_threadsafe.

IMPORTANT: with the stdio transport, stdout is the protocol channel and must not
be polluted. All logs go to a file (and stderr), never to stdout.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from typing import Any

import yaml

from .floating_window import GuardFloatingWindow
from .privacy_filter import SlackPrivacyFilter
from .slack_client import SlackClient, SlackClientError
from .slack_mcp_server import SlackGuardServer

logger = logging.getLogger("loopline.slack")

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


# ---------------------------------------------------------------------------- #
# Configuration & logging
# ---------------------------------------------------------------------------- #
def _resolve_path(path: str) -> str:
    """Resolve a possibly-relative config path against the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_config(config_path: str) -> dict[str, Any]:
    resolved = _resolve_path(config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Configuration file not found: {resolved}. "
            "Copy config/settings.yaml.example to config/settings.yaml."
        )
    with open(resolved, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {resolved} did not parse to a mapping")
    return config


def setup_logging(config: dict[str, Any]) -> None:
    """Configure file + stderr logging. Never logs to stdout (stdio transport)."""
    log_cfg = config.get("logging", {}) or {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = _resolve_path(log_cfg.get("file", "logs/slack-guard.log"))

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # stderr is safe; stdout is reserved for the MCP protocol.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    logger.info("Logging initialized at level %s -> %s", level_name, log_file)


# ---------------------------------------------------------------------------- #
# Component construction
# ---------------------------------------------------------------------------- #
def build_slack_client(config: dict[str, Any]) -> SlackClient:
    slack_cfg = config.get("slack", {}) or {}
    bot_token = slack_cfg.get("bot_token", "")
    return SlackClient(bot_token=bot_token)


def build_privacy_filter(config: dict[str, Any]) -> SlackPrivacyFilter:
    return SlackPrivacyFilter(config.get("slack_privacy", {}) or {})


# ---------------------------------------------------------------------------- #
# MCP server thread
# ---------------------------------------------------------------------------- #
class SlackMCPServerThread(threading.Thread):
    """Runs the FastMCP stdio server inside its own asyncio event loop."""

    def __init__(self, server: SlackGuardServer) -> None:
        super().__init__(name="slack-mcp-server", daemon=True)
        self._server = server

    def run(self) -> None:
        try:
            self._server.run_stdio()
        except Exception as exc:  # noqa: BLE001
            logger.error("Slack MCP server thread crashed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #
def run_app(config: dict[str, Any]) -> int:
    """Start the MCP server thread and the menu bar app (blocks until quit)."""
    try:
        slack_client = build_slack_client(config)
    except SlackClientError as exc:
        logger.error("Cannot start: %s", exc)
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1

    privacy_filter = build_privacy_filter(config)

    # Verify the bot token up front so a misconfigured token fails loudly here
    # rather than on the first Claude request.
    try:
        workspace = slack_client.check_connection()
        logger.info("Slack credentials verified for workspace %r", workspace)
    except SlackClientError as exc:
        logger.error("Cannot start: %s", exc)
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1

    mcp_cfg = config.get("mcp", {}) or {}
    server = SlackGuardServer(
        slack_client=slack_client,
        privacy_filter=privacy_filter,
        server_name=mcp_cfg.get("server_name", "slack-guard"),
        server_version=mcp_cfg.get("server_version", "0.1.0"),
    )

    server_thread = SlackMCPServerThread(server)
    server_thread.start()
    logger.info("Slack MCP server thread started")

    def _on_quit() -> None:
        logger.info("Menu bar quit handler invoked; process will exit")

    app = GuardFloatingWindow(privacy_filter=privacy_filter, on_quit=_on_quit, app_name="Slack Guard")
    logger.info("Starting floating window on main thread")
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
    return 0


# ---------------------------------------------------------------------------- #
# Argument parsing / dispatch
# ---------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="loopline-slack",
        description="macOS menu bar privacy proxy between Claude (MCP) and Slack.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the YAML config file (default: config/settings.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    setup_logging(config)

    try:
        return run_app(config)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
