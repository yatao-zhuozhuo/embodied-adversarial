"""Strict, reusable XML decision contract for baseline, SFT and GRPO callers.

No compatibility guessing or XML repair is performed. Host lifecycle, rewards
and simulator execution remain outside this module.
"""
from __future__ import annotations
import math
import re
import xml.etree.ElementTree as ET
from agent.runtime.planner import parse_xml_decision

PROTOCOL = 'openeta.two_tool_xml.v1'
PAIRS = {('tool_call', 'observe_env'), ('tool_call', 'step_env'), ('response', 'task_complete')}


def validate_action(parameters: dict, contract: dict, remaining_steps: int | None = None):
    if not isinstance(parameters, dict) or set(parameters) != {'action', 'num_steps'}:
        raise ValueError('step_env requires exactly action and num_steps')
    action, steps = parameters['action'], parameters['num_steps']
    dim = contract['action_dim']
    if not isinstance(action, list) or len(action) != dim:
        raise ValueError(f'action must be an XML array with exactly {dim} numbers')
    if len(contract['action_low']) != dim or len(contract['action_high']) != dim:
        raise ValueError('host action bounds do not match action_dim')
    for i, value in enumerate(action):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('action must contain finite numbers, not booleans')
        if not contract['action_low'][i] <= value <= contract['action_high'][i]:
            raise ValueError(f'action[{i}] outside native bounds')
    if type(steps) is not int or not 1 <= steps <= 10:
        raise ValueError('num_steps must be an integer from 1 to 10')
    if remaining_steps is not None and steps > remaining_steps:
        raise ValueError('num_steps exceeds remaining control-step budget')
    return action, steps


def parse_decision(raw: str, contract: dict, *, remaining_steps: int | None = None,
                   finish_reason: str = 'stop', allow_thinking: bool = False) -> dict:
    result = {'protocol': PROTOCOL, 'valid': False, 'errors': [], 'decision': None,
              'thinking': None, 'answer': raw, 'repair_applied': False, 'execution_request': None}
    try:
        if not isinstance(raw, str):
            raise ValueError('model reply must be XML text')
        if finish_reason != 'stop':
            raise ValueError('incomplete/truncated model reply')
        text = raw.strip()
        if text.startswith('<think>'):
            if not allow_thinking:
                raise ValueError('thinking blocks disabled in this run')
            end = text.find('</think>')
            if end < 0:
                raise ValueError('unclosed thinking block')
            result['thinking'] = text[7:end]
            text = text[end+8:].strip()
            if '<think>' in text or '</think>' in text:
                raise ValueError('multiple thinking blocks')
        result['answer'] = text
        if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper() or '<?' in text or '<!--' in text:
            raise ValueError('XML declarations, entities and comments are not permitted')
        # Parse the entire reply before invoking the shared planner parser. This
        # rejects wrappers, multiple decisions, unfinished tags and repair cases.
        root = ET.fromstring(text)
        if root.tag != 'decision' or root.attrib:
            raise ValueError('expected one unadorned decision root')
        fields = list(root)
        if len(fields) != 4 or {e.tag for e in fields} != {'kind', 'name', 'reasoning', 'parameters'}:
            raise ValueError('decision needs exactly one kind/name/reasoning/parameters field')
        if (root.text or '').strip() or any((e.tail or '').strip() for e in fields):
            raise ValueError('unexpected text outside decision fields')
        mapping = {e.tag:e for e in fields}
        for name in ('kind','name','reasoning'):
            node=mapping[name]
            if list(node) or node.attrib:
                raise ValueError(f'{name} must be plain text or CDATA')
        kind=(mapping['kind'].text or '').strip(); name=(mapping['name'].text or '').strip()
        if (kind,name) not in PAIRS:
            raise ValueError('only observe_env, step_env and task_complete are permitted')
        params=mapping['parameters']
        if params.attrib not in ({}, {'type':'object'}) or (params.text or '').strip() or any((e.tail or '').strip() for e in params):
            raise ValueError('invalid parameters container')
        if name != 'step_env':
            if list(params):raise ValueError(f'{name} requires empty parameters')
        else:
            if len(params)!=2 or {e.tag for e in params} != {'action','num_steps'}:
                raise ValueError('step_env requires exactly action and num_steps')
            action=params.find('action'); steps=params.find('num_steps')
            if action.attrib not in ({}, {'type':'array'}) or (action.text or '').strip():
                raise ValueError('action must use type="array" with numeric item children; JSON strings forbidden')
            for item in action:
                if item.tag!='item' or item.attrib not in ({}, {'type':'number'}, {'type':'integer'}) or list(item) or (item.tail or '').strip():
                    raise ValueError('action entries must be item type="number" leaves')
            if steps.attrib not in ({}, {'type':'integer'}) or list(steps) or not re.fullmatch(r'[+-]?\d+', (steps.text or '').strip()):
                raise ValueError('num_steps must use type="integer" and integer text')
        payload,errors=parse_xml_decision(text)
        if errors or payload.get('_xml_wire_repair'):
            raise ValueError('; '.join(errors) or 'XML repair forbidden')
        # Empty reasoning is free text, never a null or inferred scalar.
        payload['reasoning']=mapping['reasoning'].text or ''
        payload['kind'], payload['name'] = kind, name
        if name=='step_env':validate_action(payload['parameters'],contract,remaining_steps)
        result.update(valid=True,decision=payload)
        if kind=='tool_call':result['execution_request']={'name':name,'parameters':payload['parameters']}
    except (ValueError, TypeError, KeyError, ET.ParseError) as exc:
        result['errors']=[str(exc)]
    return result


def serialize_decision(kind: str, name: str, reasoning: str, parameters: dict, contract: dict) -> str:
    if (kind,name) not in PAIRS or not isinstance(reasoning,str):raise ValueError('invalid decision')
    if name=='step_env':validate_action(parameters,contract)
    elif parameters!={}:raise ValueError('empty parameters required')
    root=ET.Element('decision')
    for tag,value in [('kind',kind),('name',name),('reasoning',reasoning)]:ET.SubElement(root,tag).text=value
    params=ET.SubElement(root,'parameters')
    if name=='step_env':
        action=ET.SubElement(params,'action',{'type':'array'})
        for value in parameters['action']:ET.SubElement(action,'item',{'type':'number'}).text=str(value)
        ET.SubElement(params,'num_steps',{'type':'integer'}).text=str(parameters['num_steps'])
    text=ET.tostring(root,encoding='unicode')
    if not parse_decision(text,contract)['valid']:raise ValueError('serialized decision failed validation')
    return text
