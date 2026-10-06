import pytest

from agent.runtime.two_tool_xml import parse_decision, serialize_decision
from agent.runtime.embodied_two_tool import parse_command, build_prompt


@pytest.mark.parametrize("dim", [4, 7, 12])
def test_roundtrip_native_contract_and_reasoning(dim):
    parameters = {"action": [0.2] * dim, "num_steps": 3}
    text = serialize_decision("step_env", parameters, "<&> ]]> 中文\nline")
    result = parse_decision(text, action_low=[-1]*dim, action_high=[1]*dim)
    assert result["decision"]["parameters"] == parameters
    assert result["decision"]["reasoning"] == "<&> ]]> 中文\nline"
    assert result["repaired"] is False
    if dim == 7:
        assert parse_command(text, action_low=[-1]*dim, action_high=[1]*dim) == {"tool": "step_env", **parameters}


@pytest.mark.parametrize("name", ["observe_env", "task_complete"])
def test_thinking_and_empty_parameters(name):
    raw = "<think>reasoning only</think>" + serialize_decision(name)
    result = parse_decision(raw, action_low=[-1]*7, action_high=[1]*7)
    assert result["thinking"] == "reasoning only"
    assert result["decision"]["parameters"] == {}


@pytest.mark.parametrize("mutate", [
    lambda x: x+x,
    lambda x: "<think>"+x,
    lambda x: "prefix "+x,
    lambda x: x.replace("<kind>", "<name>observe_env</name><kind>"),
    lambda x: x.replace("</parameters>", "<handle>secret</handle></parameters>"),
    lambda x: x.replace('type="number">0.2', 'type="boolean">true'),
    lambda x: x.replace('type="number">0.2', 'type="number">NaN'),
    lambda x: x.replace('type="number">0.2', 'type="number">Infinity'),
    lambda x: x.replace('type="number">0.2', 'type="number">1.1'),
    lambda x: x.replace('type="integer">3', 'type="integer">3.0'),
    lambda x: x.replace('type="integer">3', 'type="integer">11'),
    lambda x: x.replace('</action>', ''),
    lambda x: x.replace('<action type="array">', '<action>'),
    lambda x: x.replace('step_env', 'create_env'),
])
def test_invalid_never_executes(mutate):
    raw = serialize_decision("step_env", {"action": [0.2]*7, "num_steps": 3})
    with pytest.raises(ValueError):
        parse_command(mutate(raw), action_low=[-1]*7, action_high=[1]*7)


def test_budget_and_provider_truncation_rejected():
    raw = serialize_decision("step_env", {"action": [0]*7, "num_steps": 3})
    for extra in ({"remaining_control_steps": 2}, {"finish_reason": "length"}):
        with pytest.raises(ValueError):
            parse_decision(raw, action_low=[-1]*7, action_high=[1]*7, **extra)
    with pytest.raises(ValueError):
        parse_command('{"tool":"observe_env"}', action_low=[-1]*7, action_high=[1]*7)


def test_prompt_is_xml_and_contains_no_hidden_state():
    prompt = build_prompt("alice", {"robot": {}}, turn=0, max_turns=32, remaining_control_steps=96)
    assert '<action type="array">' in prompt
    assert 'Return exactly one JSON' not in prompt
