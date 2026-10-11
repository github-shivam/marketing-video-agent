"""Wiring of the real agent implementations, shared by the CLI and the API."""

from __future__ import annotations

from agents.assets import analyze_node
from agents.classify import classify_node
from agents.dialogue import dialogue_node
from agents.publisher import publish_node, video_node
from agents.research_script import critic_node, research_node, writer_node
from config import Settings
from graph import Nodes


def build_nodes(cfg: Settings) -> Nodes:
    """Wire the real agent implementations into the shape the graph expects."""
    return Nodes(
        classify=classify_node(cfg),
        research=research_node(cfg),
        writer=writer_node(cfg),
        critic=critic_node(cfg),
        dialogue=dialogue_node(cfg),
        video=video_node(cfg),
        publish=publish_node(cfg),
        analyze=analyze_node(cfg),
    )
