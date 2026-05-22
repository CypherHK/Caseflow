import logging
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from schema import AgentInfo


@pytest.mark.asyncio
async def test_lifespan(monkeypatch, caplog) -> None:
    """Test that the lifespan sets up the database and store, loads the agents, and logs errors."""
    from service import service

    fake_saver_setup = False
    fake_store_setup = False

    class FakeSaver:
        async def setup(self) -> None:
            nonlocal fake_saver_setup
            fake_saver_setup = True

    class FakeStore:
        async def setup(self) -> None:
            nonlocal fake_store_setup
            fake_store_setup = True

    class FakeBuilder:
        def compile(self, checkpointer=None, store=None, name=None):
            return type(
                "ConfiguredAgent",
                (),
                {"checkpointer": checkpointer, "store": store, "name": name},
            )()

    fake_saver = FakeSaver()
    fake_store = FakeStore()

    @asynccontextmanager
    async def fake_initialize_database():
        yield fake_saver

    @asynccontextmanager
    async def fake_initialize_store():
        yield fake_store

    agents = {
        "good": type("Agent", (), {"checkpointer": None, "store": None})(),
        "bad": type("Agent", (), {"checkpointer": None, "store": None})(),
        "compiled": type(
            "CompiledAgent",
            (),
            {"checkpointer": None, "store": None, "name": "compiled", "builder": FakeBuilder()},
        )(),
    }
    configured_agents = {}

    async def fake_load_agent(agent_key: str) -> None:
        if agent_key == "bad":
            raise RuntimeError("boom")

    def fake_get_agent(agent_key: str):
        return agents[agent_key]

    def fake_set_agent_graph(agent_key: str, agent):
        configured_agents[agent_key] = agent
        agents[agent_key] = agent

    monkeypatch.setattr(service, "initialize_database", fake_initialize_database)
    monkeypatch.setattr(service, "initialize_store", fake_initialize_store)
    monkeypatch.setattr(service, "load_agent", fake_load_agent)
    monkeypatch.setattr(service, "get_agent", fake_get_agent)
    monkeypatch.setattr(service, "set_agent_graph", fake_set_agent_graph)
    monkeypatch.setattr(
        service,
        "get_all_agent_info",
        lambda: [
            AgentInfo(key="good", description=""),
            AgentInfo(key="bad", description=""),
            AgentInfo(key="compiled", description=""),
        ],
    )

    caplog.set_level(logging.INFO, logger=service.logger.name)

    async with service.lifespan(FastAPI()):
        pass

    assert fake_saver_setup
    assert fake_store_setup
    assert agents["good"].checkpointer is fake_saver
    assert agents["good"].store is fake_store
    assert agents["bad"].checkpointer is fake_saver
    assert agents["bad"].store is fake_store
    assert configured_agents["compiled"].checkpointer is fake_saver
    assert configured_agents["compiled"].store is fake_store
    assert configured_agents["compiled"].name == "compiled"

    assert "Agent loaded: good" in caplog.text
    assert "Failed to load agent bad: boom" in caplog.text
    assert "Agent loaded: compiled" in caplog.text
