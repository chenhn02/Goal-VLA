"""Extract manipulated object names from instruction text.

Uses a cascade: Gemini API -> rule-based regex -> spaCy NLP.
"""

import json
import os
import re
from typing import List


def _strip_prefix(text: str) -> str:
    t = text.strip()
    m = re.match(r"^\s*revise\s+the\s+image[\s,:;-]*", t, flags=re.IGNORECASE)
    if m:
        t = t[m.end():].lstrip()
    return t


def _extract_with_rules(text: str) -> List[str]:
    """Heuristic: extract the object the gripper directly grasps and moves.

    Tool-use instructions name the grasped object with a trailing
    'with/using the <tool>' clause (e.g. "sweep the bolts ... with the brush"),
    where the tool — not the material it acts on — is the rigid body the arm
    holds. That clause takes priority; otherwise fall back to the direct object
    of the motion verb (plain pick-and-place).
    """
    t = " ".join(text.lower().split())

    tool = re.search(r"\b(?:with|using)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+)", t)
    if tool:
        return [tool.group(1)]

    verb_pat = r"move|transfer|relocate|drag|shift|carry|bring|lift|put|place|sweep|wipe|scoop"

    m = re.search(
        rf"\b(?:{verb_pat})\b\s+(?:the\s+|a\s+|an\s+)?([^.,;:!?]+?)"
        rf"\s+\b(to|into|onto|on|in|inside|within)\b",
        t,
    )
    if m:
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", m.group(1).strip())
        if tokens:
            return [tokens[-1]]

    m2 = re.search(rf"\b(?:{verb_pat})\b\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+)", t)
    if m2:
        return [m2.group(1)]

    return []


def _extract_with_spacy(text: str) -> List[str]:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return []
    except ImportError:
        return []

    doc = nlp(text)
    seen = set()
    candidates = []

    for chunk in doc.noun_chunks:
        chunk_text = re.sub(r"^(the|a|an)\s+", "", chunk.text.strip(), flags=re.IGNORECASE)
        chunk_text = re.sub(r"[^a-zA-Z0-9_\-\s]", "", chunk_text).strip().lower()
        if chunk_text and chunk_text not in seen:
            seen.add(chunk_text)
            candidates.append(chunk_text)

    return candidates


def _extract_with_gemini(text: str) -> List[str]:
    """Use Gemini to extract moved objects as a JSON array."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []

    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return []

    system_prompt = (
        "You are an information extraction assistant for a robot manipulation "
        "pipeline. Given an English instruction, identify the SINGLE object that "
        "the robot's gripper directly GRASPS and moves as a rigid body — the "
        "object physically held in the hand.\n"
        "- In plain pick-and-place ('put/place/move/stand the X ...'), that object is X.\n"
        "- When a tool is named with a 'with the <tool>' / 'using the <tool>' clause, "
        "the grasped object is the TOOL (e.g. brush, sponge, hammer), NOT the "
        "material it acts on and NOT the destination/container.\n"
        "Return a compact JSON array with exactly ONE lowercase object noun "
        "(no adjectives, no destination/container).\n"
        "Examples:\n"
        "- 'put the tomato in the pan' -> [\"tomato\"]\n"
        "- 'stand the bottle upright' -> [\"bottle\"]\n"
        "- 'sweep the bolts into the dustpan with the brush' -> [\"brush\"]\n"
        "- 'wipe the table with the sponge' -> [\"sponge\"]\n"
        "If no graspable object can be identified, return []."
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=system_prompt + "\nInstruction: " + text)],
            )],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(getattr(resp, "text", "[]").strip())
    except Exception:
        return []


def _extract_distractors_with_gemini(text: str, target: str) -> List[str]:
    """Use Gemini to list the OTHER named objects in the instruction.

    These are used as contrastive grounding classes so the detector can assign
    each visible object to its best-matching phrase, keeping the grasped target
    separate from look-alike containers / acted-on material.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []

    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return []

    system_prompt = (
        "You are an information extraction assistant for a robot manipulation "
        "pipeline. Given an English instruction and the object the gripper grasps "
        "(the TARGET), list every OTHER concrete physical object named or clearly "
        "implied in the scene — the destination/container, the material acted on, "
        "and any nearby object. These are distractors used to disambiguate the "
        "target during detection.\n"
        "Return a compact JSON array of lowercase object nouns (no adjectives, no "
        "verbs). Do NOT include the target itself. If there are none, return [].\n"
        "Examples:\n"
        "- instruction 'sweep the bolts into the dustpan with the brush', target 'brush' -> [\"dustpan\", \"bolts\"]\n"
        "- instruction 'put the tomato in the pan', target 'tomato' -> [\"pan\"]\n"
        "- instruction 'stand the bottle upright', target 'bottle' -> []\n"
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=(
                    system_prompt + "\nInstruction: " + text + "\nTarget: " + target))],
            )],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        out = json.loads(getattr(resp, "text", "[]").strip())
        return [o for o in out if isinstance(o, str) and o.strip().lower() != target.lower()]
    except Exception:
        return []


def _extract_distractors_with_rules(text: str, target: str) -> List[str]:
    """Regex fallback: the 'into/onto/in the <container>' object and, for
    tool-use instructions, the acted-on direct object."""
    t = " ".join(text.lower().split())
    out: List[str] = []

    dest = re.search(r"\b(?:into|onto|on|in|inside|within)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+)", t)
    if dest:
        out.append(dest.group(1))

    # 'sweep the <material> ... with the <tool>' -> material is a distractor
    verb_pat = r"move|transfer|relocate|drag|shift|carry|bring|lift|put|place|sweep|wipe|scoop"
    mat = re.search(rf"\b(?:{verb_pat})\b\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+)", t)
    if mat and re.search(r"\b(?:with|using)\b", t):
        out.append(mat.group(1))

    seen = set()
    return [o for o in out if o != target and not (o in seen or seen.add(o))]


def extract_objects(text: str) -> List[str]:
    """Extract manipulated object names from an instruction.

    Tries Gemini first, then rule-based, then spaCy as fallback.
    """
    t = _strip_prefix(text)

    result = _extract_with_gemini(t)
    if result:
        return result

    result = _extract_with_rules(t)
    if result:
        return result

    nouns = _extract_with_spacy(t)
    return nouns[:1] if nouns else []


def extract_scene_objects(text: str) -> dict:
    """Extract the grasped target plus contrastive distractor objects.

    Returns {"target": <str|None>, "distractors": [<str>, ...]}. The distractor
    list always includes "robot arm" so the detector can bind the manipulator to
    its own phrase instead of over-grounding the target onto it. Callers should
    ground on target + distractors but keep only masks labelled with the target.
    """
    t = _strip_prefix(text)
    targets = extract_objects(t)
    target = targets[0] if targets else None

    distractors: List[str] = []
    if target:
        distractors = _extract_distractors_with_gemini(t, target)
        if not distractors:
            distractors = _extract_distractors_with_rules(t, target)

    # de-dup, drop any that collide with the target, always add the manipulator
    seen = set()
    clean = []
    for d in distractors + ["robot arm"]:
        d = d.strip().lower()
        if d and d != (target or "").lower() and d not in seen:
            seen.add(d)
            clean.append(d)

    return {"target": target, "distractors": clean}
