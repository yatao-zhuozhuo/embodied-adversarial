import pytest

from agent.runtime.planner_xml import parse_xml_decision
from agent.runtime.two_tool_xml import action_contract, parse_decision, serialize_decision


def contract(dimension=7):
    return {"action_dim": dimension, "action_low": [-1] * dimension, "action_high": [1] * dimension}


@pytest.mark.parametrize("dimension", [4, 7, 12])
def test_round_trip_and_planner_compatibility(dimension):
    parameters = {"action": [0.2] * dimension, "num_steps": 3}
    text = serialize_decision("step_env", parameters, "a < b & c ]]>\nnext")
    parsed = parse_decision(text, contract=contract(dimension), remaining_steps=3)
    generic, errors = parse_xml_decision(text)
    assert not errors
    assert parsed["decision"] == generic
    assert generic["parameters"] == parameters
    assert generic["reasoning"] == "a < b & c ]]>\nnext"


@pytest.mark.parametrize("kind,name", [("tool_call", "observe_env"), ("response", "task_complete")])
def test_empty_parameters_and_closed_thinking(kind, name):
    answer = serialize_decision(name, {})
    parsed = parse_decision("<think>do not execute <decision>here</decision></think>\n" + answer,
                            contract=contract(), remaining_steps=0)
    assert parsed["decision"]["parameters"] == {}
    assert parsed["answer"] == answer
    assert parsed["repaired"] is False


@pytest.mark.parametrize("transform", [
    lambda text: text + text,
    lambda text: "```xml\n" + text + "\n```",
    lambda text: "<think>" + text,
    lambda text: text.replace("</decision>", "<name>step_env</name></decision>"),
    lambda text: text.replace('<action type="array">', '<action type="array"><item type="boolean">true</item>'),
    lambda text: text.replace('<item type="number">0.0</item>', '<item type="number">NaN</item>', 1),
    lambda text: text.replace('<item type="number">0.0</item>', '<item type="number">Infinity</item>', 1),
    lambda text: text.replace('<item type="number">0.0</item>', '<item type="number">2</item>', 1),
    lambda text: text.replace('<num_steps type="integer">1</num_steps>', '<num_steps type="number">1.0</num_steps>'),
    lambda text: text.replace('<num_steps type="integer">1</num_steps>', '<num_steps type="integer">11</num_steps>'),
    lambda text: text.replace('</action>', ''),
    lambda text: text.replace('</parameters>', '<handle>host</handle></parameters>'),
    lambda text: text.replace('step_env', 'create_env'),
    lambda text: text.replace('tool_call', 'tool_batch'),
    lambda text: text.replace('<reasoning />', '<reasoning><decision /></reasoning>'),
])
def test_invalid_decisions_are_rejected(transform):
    text = serialize_decision("step_env", {"action": [0.0] * 7, "num_steps": 1})
    with pytest.raises(ValueError):
        parse_decision(transform(text), contract=contract(), remaining_steps=10)


def test_budget_dimension_and_array_string_rejected():
    text = serialize_decision("step_env", {"action": [0] * 7, "num_steps": 2})
    for action_spec, remaining in [(contract(4), 10), (contract(), 1)]:
        with pytest.raises(ValueError):
            parse_decision(text, contract=action_spec, remaining_steps=remaining)
    with pytest.raises(ValueError):
        parse_decision('<decision><kind>tool_call</kind><name>step_env</name><reasoning/><parameters><action>[0,0,0,0,0,0,0]</action><num_steps type="integer">1</num_steps></parameters></decision>', contract=contract(), remaining_steps=10)
    with pytest.raises(ValueError):
        action_contract({"action_dim": 4})
