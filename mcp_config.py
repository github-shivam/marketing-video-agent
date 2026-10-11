"""MCP server configuration and per-agent tool loading.

Design rules:

* A server whose credential is missing is skipped silently — the agent that
  wanted it must still work with zero tools.
* A server that fails to start logs a warning and returns no tools. One bad
  server never takes down the run.
* ``PATH`` is forwarded into every stdio server's environment so that ``npx``
  and ``uvx`` resolve, and on Windows the ``.cmd`` shim is located explicitly
  because ``CreateProcess`` will not find a bare ``npx``.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)

BRIGHTDATA = "brightdata"


def _resolve_command(command: str) -> str:
    """Return an executable path for ``command``.

    On Windows ``npx`` is really ``npx.cmd``; passing the bare name to a
    non-shell subprocess raises FileNotFoundError.
    """
    return shutil.which(command) or command


def _stdio_env(extra: dict[str, str]) -> dict[str, str]:
    """Build a stdio server environment, always including PATH."""
    env = {"PATH": os.environ.get("PATH", "")}
    env.update({k: v for k, v in extra.items() if v})
    return env


def build_server_configs(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return ``MultiServerMCPClient`` configs for every server we have a key for.

    Args:
        settings: Supplies the credentials and tuning for each server.

    Returns:
        A mapping of server name to connection config. Servers with no
        credential are omitted entirely.
    """
    servers: dict[str, dict[str, Any]] = {}

    if settings.brightdata_api_token:
        servers[BRIGHTDATA] = {
            "transport": "stdio",
            "command": _resolve_command("npx"),
            "args": ["-y", "@brightdata/mcp"],
            # TOOLS names individual tools and takes precedence; GROUPS enables
            # whole categories. Unset, the server exposes 29 tools, most of
            # them irrelevant here (LinkedIn, Reddit, TikTok Shop...), which
            # only degrades the research agent's tool choice.
            "env": _stdio_env(
                {
                    "API_TOKEN": settings.brightdata_api_token,
                    "TOOLS": settings.brightdata_tools,
                    "GROUPS": (
                        "" if settings.brightdata_tools else settings.brightdata_groups
                    ),
                    "RATE_LIMIT": settings.brightdata_rate_limit,
                    "BASE_MAX_RETRIES": str(settings.brightdata_max_retries),
                }
            ),
        }
    else:
        logger.info("BRIGHTDATA_API_TOKEN not set - research will run tool-free")

    return servers


async def load_tools(settings: Settings, server_name: str) -> list[Any]:
    """Load the tools for one MCP server.

    Each agent calls this with its own server name so that it only ever sees
    its own tools. A missing or broken server yields an empty list.

    Args:
        settings: Supplies the server configs.
        server_name: Which server's tools to load.

    Returns:
        The server's LangChain tools, or ``[]`` if unavailable.
    """
    servers = build_server_configs(settings)
    if server_name not in servers:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(servers)
        tools = await client.get_tools(server_name=server_name)
        logger.info("Loaded %d tool(s) from MCP server %r", len(tools), server_name)
        return tools
    except Exception as exc:  # noqa: BLE001 - a bad server must not end the run
        logger.warning(
            "MCP server %r failed to load (%s: %s) — continuing without its tools",
            server_name,
            type(exc).__name__,
            exc,
        )
        return []
