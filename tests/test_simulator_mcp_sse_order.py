"""SSE schema lookup must finish before a simulator action can mutate state."""
import asyncio
from contextlib import asynccontextmanager
import sys
from types import ModuleType

import pytest
from agent.tools.sim_mcp import _call_sse_mcp_tool


@pytest.mark.parametrize('failure', [None, 'listing', 'validation'])
def test_schema_lookup_precedes_action_and_actions_are_never_replayed(monkeypatch, failure):
    events = []
    class Session:
        def __init__(self, *args):
            self.cached = False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            events.append('initialize')
        async def list_tools(self):
            # Model the observed failure: metadata transport after an action
            # can fail even though the simulator already executed that action.
            assert 'step_env' not in events
            events.append('list_tools')
            if failure == 'listing':
                raise TimeoutError('metadata unavailable')
            self.cached = True
        async def call_tool(self, name, arguments, **kwargs):
            events.append(name)
            if not self.cached:
                await self.list_tools()
            if failure == 'validation':
                raise RuntimeError('Invalid structured content')
            return {'structuredContent': {'executed_control_steps': 1}, 'isError': False}
    @asynccontextmanager
    async def connect(url, **kwargs):
        assert kwargs['headers']['Connection'] == 'close'
        yield None, None
    mcp = ModuleType('mcp'); mcp.ClientSession = Session
    sse = ModuleType('mcp.client.sse'); sse.sse_client = connect
    monkeypatch.setitem(sys.modules, 'mcp', mcp)
    monkeypatch.setitem(sys.modules, 'mcp.client.sse', sse)
    call = _call_sse_mcp_tool(url='http://127.0.0.1/sse', tool_name='step_env', arguments={}, timeout_s=1)
    if failure:
        with pytest.raises(TimeoutError if failure == 'listing' else RuntimeError):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == {'executed_control_steps': 1}
    assert events == ['initialize', 'list_tools'] + ([] if failure == 'listing' else ['step_env'])
