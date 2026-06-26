"""各数据源原始格式 -> 统一中间表示 (IR) 的适配器。

IR 规范见 docs/data-format.md。每个 adapter 输入一条原始 row(dict)，
输出一条 IR(dict) 或 None(无法解析/应丢弃)。

只用标准库，便于离线自测。
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 通用 helper
# --------------------------------------------------------------------------- #


_TYPE_MAP = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer", "long": "integer",
    "float": "number", "number": "number", "double": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array",
    "dict": "object", "object": "object",
}


def _normalize_type(t: Any) -> Any:
    """把 Python/非标准类型名归一到 JSON Schema 类型。未知的保留原样。"""
    if not isinstance(t, str):
        return t
    low = t.strip().lower()
    if low in _TYPE_MAP:
        return _TYPE_MAP[low]
    if low.startswith("list[") or low.startswith("array"):
        return "array"
    if low.startswith("dict") or low.startswith("object"):
        return "object"
    return t


def _normalize_prop(v: Any) -> Dict[str, Any]:
    if not isinstance(v, dict):
        return {"type": "string"}
    out = dict(v)
    if "type" in out:
        out["type"] = _normalize_type(out["type"])
    return out


def _normalize_params(params: Any) -> Dict[str, Any]:
    """把三种风格的 parameters 统一成标准 JSON Schema:
      1) 标准: {"type":"object","properties":{...},"required":[...]}
      2) ToolACE: 同上但 type=="dict"
      3) xLAM 扁平: {"pname": {"description":...,"type":"str"}, ...}(无 properties 包裹)
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "required": []}
    if "properties" in params:  # 风格 1 / 3-with-properties
        out = dict(params)
        if out.get("type") in ("dict", None):
            out["type"] = "object"
        out["properties"] = {k: _normalize_prop(v) for k, v in (out.get("properties") or {}).items()}
        out.setdefault("required", [])
        return out
    # 风格 3: 扁平参数字典(值是 {description,type,...} 形态)
    props: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict) and ("type" in v or "description" in v):
            props[k] = _normalize_prop(v)
    required = [
        k for k, v in params.items()
        if isinstance(v, dict) and "default" not in v and not v.get("optional")
        and ("type" in v or "description" in v)
    ]
    return {"type": "object", "properties": props, "required": required}


def _coerce_tool_schema(obj: Any) -> Optional[Dict[str, Any]]:
    """把单个工具/函数定义规整为标准 {name, description, parameters(JSON Schema)}。"""
    if not isinstance(obj, dict):
        return None
    # OpenAI 可能包成 {"type":"function","function":{...}}
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not name:
        return None
    return {
        "name": name,
        "description": obj.get("description", "") or "",
        "parameters": _normalize_params(obj.get("parameters")),
    }


def _coerce_tools(raw: Any) -> List[Dict[str, Any]]:
    """raw 可能是 list / dict / JSON 字符串。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            t = _coerce_tool_schema(item)
            if t:
                out.append(t)
    return out


def _parse_args(raw: Any) -> Dict[str, Any]:
    """arguments 可能是 dict，或 JSON 字符串，甚至二次编码(字符串里又是 JSON)。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            return {"_raw": s}
        if isinstance(v, str):  # 二次编码
            try:
                v2 = json.loads(v)
                return v2 if isinstance(v2, dict) else {"_value": v2}
            except json.JSONDecodeError:
                return {"_raw": v}
        return v if isinstance(v, dict) else {"_value": v}
    return {}


def _extract_json_objects(text: str) -> List[Any]:
    """从一段文本里扫出所有顶层 {...} JSON 对象(括号配平，跳过字符串内的括号)。"""
    objs: List[Any] = []
    for span in _iter_brace_spans(text):
        try:
            objs.append(json.loads(span))
        except json.JSONDecodeError:
            pass
    return objs


def _iter_brace_spans(text: str) -> List[str]:
    """返回所有顶层配平的 {...} 子串(只把双引号当字符串边界，单引号不算，
    这样 glaive 那种 arguments 用单引号包裹、内部还有花括号的也能正确配平)。"""
    spans: List[str] = []
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start : i + 1])
                    start = None
    return spans


# --------------------------------------------------------------------------- #
# Hermes 风格 conversations 解析（NousResearch/hermes-function-calling-v1、
# interstellarninja/hermes_reasoning_tool_use 都走这条；ToolACE 也复用）
#   - system 里 <tools>[...]</tools> 放工具定义
#   - assistant 里 <tool_call>{...}</tool_call> 发起调用
#   - tool 角色里 <tool_response>{...}</tool_response> 返回结果
# --------------------------------------------------------------------------- #

_TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FC_MODEL_ANCHOR = "You are a function calling AI model"


def _strip_think_text(text: str) -> str:
    """去掉 <think>...</think>;并处理未闭合的 <think>(删到末尾)与残留标签。"""
    text = _THINK_RE.sub("", text)
    i = text.find("<think>")  # 未闭合 -> 删到末尾
    if i != -1:
        text = text[:i]
    return text.replace("</think>", "").replace("<think>", "")


def _strip_think_system(sys_text: str) -> str:
    """strip_think 时清掉 system 里的'深度思考'前言(从 'function calling AI model' 起保留)。"""
    a = sys_text.find(_FC_MODEL_ANCHOR)
    if a > 0:
        sys_text = sys_text[a:]
    return _strip_think_text(sys_text).strip()


def _tools_from_system(system_text: str) -> List[Dict[str, Any]]:
    # Hermes 的 system 通常先有一句 "...within <tools></tools> XML tags." 的说明(里面是空的
    # <tools></tools>)，再跟真正的 <tools>[...]</tools>。所以遍历所有匹配，取第一个有内容的。
    for m in _TOOLS_RE.finditer(system_text or ""):
        inner = m.group(1).strip()
        if not inner:
            continue
        tools = _coerce_tools(inner)
        if not tools:  # 兜底：内部可能是逐个 JSON 对象而非 JSON 数组
            tools = [t for o in _extract_json_objects(inner) if (t := _coerce_tool_schema(o))]
        if tools:
            return tools
    return []


def _clean_system(system_text: Optional[str]) -> Optional[str]:
    """去掉 <tools> 块后剩余的 system 文本。"""
    if not system_text:
        return None
    t = _TOOLS_RE.sub("", system_text).strip()
    return t or None


def adapt_hermes_conversations(
    row: Dict[str, Any], source: str, strip_think: bool = False
) -> Optional[Dict[str, Any]]:
    convs = row.get("conversations")
    if not isinstance(convs, list):
        convs = row.get("messages")
    if not isinstance(convs, list):
        return None

    tools = _coerce_tools(row.get("tools"))  # 可能有独立 tools 列
    system_text: Optional[str] = None
    turns: List[Dict[str, Any]] = []

    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value")
        if val is None:
            val = turn.get("content")
        if val is None:
            continue
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)

        if role == "system":
            if not tools:
                tools = _tools_from_system(val)
            system_text = _clean_system(val)
            if strip_think and system_text:
                system_text = _strip_think_system(system_text)
        elif role in ("human", "user"):
            if val.strip():
                turns.append({"role": "user", "content": val.strip()})
        elif role in ("gpt", "assistant", "model"):
            calls: List[Dict[str, Any]] = []
            for cm in _TOOLCALL_RE.finditer(val):
                try:
                    obj = json.loads(cm.group(1).strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("name"):
                    calls.append(
                        {"name": obj["name"], "arguments": _parse_args(obj.get("arguments", {}))}
                    )
            text = _TOOLCALL_RE.sub("", val)
            if strip_think:
                text = _strip_think_text(text)
            text = text.strip()
            if text:  # 解释/推理文本（放调用之前）
                turns.append({"role": "assistant", "content": text})
            if calls:
                turns.append({"role": "tool_calls", "calls": calls})
        elif role in ("tool", "observation", "function", "tool_response"):
            resp = _TOOLRESP_RE.findall(val)  # 并行时一个 tool 轮可能含多个 <tool_response>
            content = "\n".join(r.strip() for r in resp) if resp else val.strip()
            if content:
                turns.append({"role": "tool", "content": content})

    if not turns:
        return None
    sc = (row.get("scenario_category") or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "source": source,
        "system": system_text,
        "tools": tools,
        "turns": turns,
        "category": sc or "unknown",  # 数据集自带的 BFCL 风格类别(没有则后续结构推断)
    }


# --------------------------------------------------------------------------- #
# glaiveai/glaive-function-calling-v2
#   columns: system(str) + chat(str)
#   chat 形如:  USER: ...  ASSISTANT: ... <functioncall> {json} <|endoftext|>
#               FUNCTION RESPONSE: {json}  ASSISTANT: ...
# --------------------------------------------------------------------------- #

_GLAIVE_SPLIT_RE = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
_GLAIVE_SQ_ARGS_RE = re.compile(r":\s*'(\{.*\})'", re.DOTALL)
_FUNCCALL_TAG = "<functioncall>"
_EOT = "<|endoftext|>"


def _glaive_loads(s: str) -> Optional[Dict[str, Any]]:
    """解析 glaive 的 functioncall 对象；它常把 arguments 写成单引号包裹的 JSON 串。"""
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # 把  "arguments": '{...}'  改写成合法的 JSON 字符串(再由 _parse_args 二次解码)
    s2 = _GLAIVE_SQ_ARGS_RE.sub(lambda m: ": " + json.dumps(m.group(1), ensure_ascii=False), s)
    try:
        v = json.loads(s2)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def _glaive_segments(chat: str) -> List[tuple]:
    parts = _GLAIVE_SPLIT_RE.split(chat)
    segs: List[tuple] = []
    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        segs.append((marker, text))
        i += 2
    return segs


def adapt_glaive(row: Dict[str, Any], source: str = "glaive") -> Optional[Dict[str, Any]]:
    chat = row.get("chat") or ""
    if not chat:
        return None
    # 工具定义在 system 文本里(若干 JSON 对象)
    tools = [t for o in _extract_json_objects(row.get("system") or "") if (t := _coerce_tool_schema(o))]

    turns: List[Dict[str, Any]] = []
    for marker, text in _glaive_segments(chat):
        text = text.strip()
        if not text:
            continue
        if marker == "USER:":
            clean = text.replace(_EOT, "").strip()
            if clean:
                turns.append({"role": "user", "content": clean})
        elif marker == "ASSISTANT:":
            if _FUNCCALL_TAG in text:
                pre = text[: text.index(_FUNCCALL_TAG)].replace(_EOT, "").strip()
                if pre:
                    turns.append({"role": "assistant", "content": pre})
                rest = text[text.index(_FUNCCALL_TAG) + len(_FUNCCALL_TAG) :]
                spans = _iter_brace_spans(rest)
                obj = _glaive_loads(spans[0]) if spans else None
                if obj and obj.get("name"):
                    turns.append(
                        {
                            "role": "tool_calls",
                            "calls": [
                                {"name": obj["name"], "arguments": _parse_args(obj.get("arguments", {}))}
                            ],
                        }
                    )
                post = rest.split(_EOT, 1)[1].strip() if _EOT in rest else ""
                if post:
                    turns.append({"role": "assistant", "content": post})
            else:
                clean = text.replace(_EOT, "").strip()
                if clean:
                    turns.append({"role": "assistant", "content": clean})
        elif marker == "FUNCTION RESPONSE:":
            clean = text.replace(_EOT, "").strip()
            if clean:
                turns.append({"role": "tool", "content": clean})

    if not turns:
        return None
    return {"source": source, "system": None, "tools": tools, "turns": turns, "category": "unknown"}


# --------------------------------------------------------------------------- #
# Team-ACE/ToolACE  —— assistant 工具调用是 Python 伪调用串 [Func(arg="v")]，
# 工具定义是 system 文本里的 JSON 数组(参数用 type:dict)。专门解析。
# --------------------------------------------------------------------------- #


def _first_json_array_str(text: str) -> Optional[str]:
    """返回第一个配平的 [...] 子串(双引号当字符串边界)。"""
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
    return None


def _split_top_commas(s: str) -> List[str]:
    """按顶层逗号切分(尊重 ()[]{} 嵌套与引号)。"""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str = False
    q = ""
    esc = False
    for ch in s:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            q = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _lit(s: str) -> Any:
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s.strip().strip('"').strip("'")  # 兜底当裸字符串


def _parse_toolace_args(argstr: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for piece in _split_top_commas(argstr):
        if "=" not in piece:
            continue
        key, _, valuestr = piece.partition("=")
        out[key.strip()] = _lit(valuestr)
    return out


def _parse_toolace_calls(s: str, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """解析 [Func(arg="v"), Func2(...)]。ToolACE 函数名可能自带括号(如
    'User Feed (Video Posts) V2')，所以优先用已知工具名做最长前缀匹配来锚定名字，
    匹配不到再退回"到第一个 ( 为止"。"""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    names_sorted = sorted(tool_names or [], key=len, reverse=True)
    calls: List[Dict[str, Any]] = []
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in ", \t\n":
            i += 1
        if i >= n:
            break
        name: Optional[str] = None
        nameend = -1
        for tn in names_sorted:  # 已知工具名前缀匹配(后跟 '(')
            if s.startswith(tn, i):
                k = i + len(tn)
                while k < n and s[k] in " \t":
                    k += 1
                if k < n and s[k] == "(":
                    name, nameend = tn, k
                    break
        if name is None:  # 退路：名字取到第一个 '('
            j = s.find("(", i)
            if j == -1:
                break
            name, nameend = s[i:j].strip(), j
        # 从 nameend 的 '(' 起配平
        depth = 0
        m = nameend
        in_str = False
        q = ""
        esc = False
        while m < n:
            ch = s[m]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == q:
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str = True
                    q = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            m += 1
        if name:
            calls.append({"name": name, "arguments": _parse_toolace_args(s[nameend + 1 : m])})
        i = m + 1
    return calls


def adapt_toolace(row: Dict[str, Any], source: str = "toolace") -> Optional[Dict[str, Any]]:
    """ToolACE: system 里有 JSON 函数列表; assistant 工具调用是 Python 伪调用串
    形如 [FuncName(arg="v", n=3)]，不是 <tool_call> JSON。需专门解析。"""
    convs = row.get("conversations")
    if not isinstance(convs, list):
        return None
    sys_text = row.get("system") or ""
    if not sys_text:
        for t in convs:
            if isinstance(t, dict) and (t.get("from") or t.get("role")) == "system":
                sys_text = t.get("value") or t.get("content") or ""
                break
    tools: List[Dict[str, Any]] = []
    arr = _first_json_array_str(sys_text)
    if arr:
        try:
            tools = [t for o in json.loads(arr) if (t := _coerce_tool_schema(o))]
        except json.JSONDecodeError:
            tools = []

    turns: List[Dict[str, Any]] = []
    for t in convs:
        if not isinstance(t, dict):
            continue
        role = (t.get("from") or t.get("role") or "").lower()
        val = t.get("value")
        if val is None:
            val = t.get("content")
        if val is None:
            continue
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        v = val.strip()
        if role == "system":
            continue  # 工具已单列
        if role in ("user", "human"):
            if v:
                turns.append({"role": "user", "content": v})
        elif role in ("assistant", "gpt"):
            calls = (
                _parse_toolace_calls(v, [t["name"] for t in tools])
                if (v.startswith("[") and v.endswith("]") and "(" in v)
                else []
            )
            if calls:
                turns.append({"role": "tool_calls", "calls": calls})
            elif v:
                turns.append({"role": "assistant", "content": v})  # 自然语言回答/拒答
        elif role in ("tool", "observation", "function"):
            if v:
                turns.append({"role": "tool", "content": v})

    if not turns:
        return None
    return {"source": source, "system": None, "tools": tools, "turns": turns, "category": "unknown"}


ADAPTERS = {
    "hermes": adapt_hermes_conversations,
    "glaive": adapt_glaive,
    "toolace": adapt_toolace,
}
