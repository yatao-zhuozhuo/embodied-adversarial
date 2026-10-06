"""Public XML decoding shared with the general OpenETA planner."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from typing import Any
JsonDict = dict[str, Any]

# Leaf element names whose text must remain verbatim. This prevents identifiers
# and code-like text from being inferred as numbers merely because they happen
# to contain a numeric-looking value.
_XML_STRING_LEAF_NAMES = frozenset(
    {
        "camera_frame_id",
        "code",
        "kind",
        "message",
        "name",
        "prompt",
        "query",
        "reasoning",
        "skill",
        "source_packet_id",
        "target_mask",
        "tool",
        "url",
    }
)

# These child names are the explicit list vocabulary used by the planner wire
# contract. Other repeated child tags are objects and are rejected downstream
# if they do not match the requested ToolContract schema.
_XML_LIST_ITEM_NAMES = frozenset({"item", "call"})

# Some single-item arrays are naturally emitted as an XML object without an
# extra ``<item>`` wrapper. Keep this vocabulary deliberately narrow: these
# are wire-level collection fields whose element object is unambiguous. The
# downstream ToolContract validator remains the authority for the contents.
_XML_SINGLE_OBJECT_ARRAY_NAMES = frozenset({"points"})

_XML_TOKEN_PATTERN = re.compile(
    r"<!\[CDATA\[.*?\]\]>|<!--.*?-->|<[^>]+>",
    re.DOTALL,
)
_XML_OPEN_TAG_PATTERN = re.compile(r"<\s*([A-Za-z_][\w.:-]*)\b[^>]*>", re.DOTALL)
_XML_CLOSE_TAG_PATTERN = re.compile(r"</\s*([A-Za-z_][\w.:-]*)\s*>")


def parse_xml_decision(payload: str) -> tuple[JsonDict, list[str]]:
    """Public XML decoder; execution policies must validate the decoded decision."""
    return _parse_xml_decision(payload)


def parse_xml_decision(payload: str) -> tuple[JsonDict, list[str]]:
    """Parse one main-planner ``<decision>`` into the existing decision shape.

    Host-generated failure payloads remain dictionaries and bypass this parser.
    Main-model strings are intentionally XML-only so a malformed or stale JSON
    response is visible to the existing validation-retry loop instead of being
    silently accepted through a compatibility path.
    """

    text = _strip_code_fence(payload)
    start = text.find("<decision")
    end = text.rfind("</decision>")
    if start == -1 or end == -1:
        return {}, ["Planner backend returned text without a <decision> element."]
    document = text[start : end + len("</decision>")]
    repair_tags: list[str] = []
    try:
        root = ET.fromstring(document)
    except ET.ParseError as original_exc:
        repaired_document, repair_tags = _repair_unclosed_xml_list_containers(document)
        if not repair_tags:
            return {}, [f"Planner backend returned invalid XML: {original_exc}"]
        try:
            root = ET.fromstring(repaired_document)
        except ET.ParseError:
            return {}, [f"Planner backend returned invalid XML: {original_exc}"]
    if root.tag != "decision":
        return {}, [f"Planner backend XML root must be <decision>, got <{root.tag}>."]

    value = _xml_element_to_value(root)
    if not isinstance(value, dict):
        return {}, ["Planner backend <decision> must decode to an object."]
    if repair_tags:
        value["_xml_wire_repair"] = {
            "schema_version": "openeta.planner_xml_repair.v1",
            "kind": "close_unclosed_list_container",
            "inserted_closing_tags": repair_tags,
            "count": len(repair_tags),
        }
    return value, []


def _repair_unclosed_xml_list_containers(document: str) -> tuple[str, list[str]]:
    """Close only list containers whose omitted end tag is unambiguous.

    The XML wire contract defines a container with ``<item>`` or ``<call>``
    children as a list.  Models occasionally emit a complete list and begin a
    sibling field without closing that container, for example ``<xyz><item>``
    followed by ``<quat_xyzw>``.  Re-parenting the sibling by inserting
    ``</xyz>`` is deterministic.  No object element, scalar, or arbitrary tag
    mismatch is repaired here; those remain visible to the validation retry
    loop.
    """

    output: list[str] = []
    stack: list[dict[str, object]] = []
    repaired: list[str] = []
    cursor = 0

    def close_intervening_lists(next_name: str) -> None:
        while stack and stack[-1]["is_list"] is True and next_name not in _XML_LIST_ITEM_NAMES:
            name = str(stack.pop()["name"])
            output.append(f"</{name}>")
            repaired.append(name)

    for match in _XML_TOKEN_PATTERN.finditer(document):
        output.append(document[cursor : match.start()])
        token = match.group(0)
        cursor = match.end()

        if token.startswith("<![CDATA[") or token.startswith("<!--"):
            output.append(token)
            continue
        if token.startswith("<?") or token.startswith("<!"):
            output.append(token)
            continue

        closing = _XML_CLOSE_TAG_PATTERN.fullmatch(token)
        if closing is not None:
            name = closing.group(1)
            while stack and str(stack[-1]["name"]) != name:
                if stack[-1]["is_list"] is not True:
                    break
                missing = str(stack.pop()["name"])
                output.append(f"</{missing}>")
                repaired.append(missing)
            output.append(token)
            if stack and str(stack[-1]["name"]) == name:
                stack.pop()
            continue

        opening = _XML_OPEN_TAG_PATTERN.fullmatch(token)
        if opening is None:
            output.append(token)
            continue
        name = opening.group(1)
        close_intervening_lists(name)
        output.append(token)
        if stack and name in _XML_LIST_ITEM_NAMES:
            stack[-1]["is_list"] = True
        if not token.rstrip().endswith("/>"):
            stack.append({"name": name, "is_list": False})

    output.append(document[cursor:])
    if not repaired:
        return document, []
    return "".join(output), repaired


def _xml_element_to_value(element: ET.Element) -> object:
    """Convert the XML wire vocabulary into JSON-equivalent Python values."""

    declared = (element.get("type") or "").strip().lower()
    children = list(element)
    if not children:
        if declared == "object":
            return {}
        if declared == "array":
            return []
        if element.tag == "parameters" and (element.text or "").strip() == "":
            return {}
        return _coerce_xml_scalar(element)

    child_tags = {child.tag for child in children}
    if declared == "array" or child_tags & _XML_LIST_ITEM_NAMES:
        return [_xml_element_to_value(child) for child in children]
    if element.tag in _XML_SINGLE_OBJECT_ARRAY_NAMES:
        if len(child_tags) == 1 and next(iter(child_tags)) in {"point"}:
            return [_xml_element_to_value(child) for child in children]
        return [
            {child.tag: _xml_element_to_value(child) for child in children}
        ]

    result: JsonDict = {}
    for child in children:
        result[child.tag] = _xml_element_to_value(child)
    return result


def _coerce_xml_scalar(element: ET.Element) -> object:
    """Coerce one XML leaf while preserving free text and CDATA verbatim.

    Explicit XML ``type`` attributes remain authoritative, while the natural
    scalar spellings ``true``, ``false``, and ``null`` are also inferred. This
    keeps planner XML equivalent to the former JSON wire format when a model
    omits a redundant type attribute. Reserved free-text fields are checked
    first and therefore remain strings even when they contain those words.
    """

    raw = element.text if element.text is not None else ""
    declared = (element.get("type") or "").strip().lower()
    if declared == "null":
        return None
    if declared == "boolean":
        return raw.strip().lower() in {"true", "1", "yes"}
    if declared == "string":
        return raw
    if declared == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    if declared == "number":
        try:
            return float(raw.strip())
        except ValueError:
            return raw
    if element.tag in _XML_STRING_LEAF_NAMES:
        return raw

    stripped = raw.strip()
    if stripped == "":
        return raw
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return raw


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped
