from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AgentContext:
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, context: AgentContext, output_root: Path) -> AgentContext:
        """Override in subclasses. Should perform data/model/explain and
        return an updated context."""
        raise NotImplementedError()


class Coordinator:
    def __init__(self, name: str, agents: List[BaseAgent], output_root: Path):
        self.name = name
        self.agents = agents
        self.output_root = output_root

    def run(self) -> AgentContext:
        ctx = AgentContext()
        errors = []
        for agent in self.agents:
            try:
                ctx = agent.run(ctx, self.output_root)
                ctx.data.setdefault('agent_used', agent.name)
                return ctx
            except Exception as exc:  # noqa: BLE001
                errors.append((agent.name, str(exc)))
        # if all failed, raise with combined message
        raise RuntimeError(f'All agents failed: {errors}')
