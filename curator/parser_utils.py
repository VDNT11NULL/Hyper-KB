import json
import re

def parse_json_safely(text: str, validate_fn, fill_defaults_fn):
    try:
        parsed = json.loads(text)
        if validate_fn(parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Extract JSON block patterns (same logic)
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*\{[^{}]*\}[^{}]*\})',
        r'(\{[^{}]+\})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if validate_fn(parsed):
                    return parsed
            except:
                continue

    # Manual key extraction fallback
    try:
        result = {}
        kw_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if kw_match:
            kw_str = kw_match.group(1)
            keywords = [k.strip(' "\'') for k in kw_str.split(',') if k.strip()]
            result['keywords'] = keywords

        ctx_match = re.search(r'"context_passage"\s*:\s*"([^"]*)"', text, re.DOTALL)
        if ctx_match:
            result['context_passage'] = ctx_match.group(1)

        entities = {"PERSON": [], "ORG": [], "DATE": [], "MISC": []}
        ent_match = re.search(r'"entities"\s*:\s*\{(.*?)\}', text, re.DOTALL)
        if ent_match:
            ent_content = ent_match.group(1)
            for etype in entities.keys():
                type_match = re.search(rf'"{etype}"\s*:\s*\[(.*?)\]', ent_content, re.DOTALL)
                if type_match:
                    items = [
                        item.strip(' "\'')
                        for item in type_match.group(1).split(',')
                        if item.strip()
                    ]
                    entities[etype] = items

        result['entities'] = entities
        if result:
            return fill_defaults_fn(result)

    except Exception as e:
        print(f"⚠️ Manual extraction failed: {e}")

    return {}

def fix_entity_classification(data: dict) -> dict:
    entities = data.get("entities", {})
    person_indicators = ["et al.", "dr.", "prof.", "mr.", "ms.", "mrs."]
    misc_items = entities.get("MISC", [])
    persons = entities.get("PERSON", [])

    to_move = [m for m in misc_items if any(ind in m.lower() for ind in person_indicators)]
    entities["MISC"] = [m for m in misc_items if m not in to_move]
    entities["PERSON"] = persons + to_move
    data["entities"] = entities
    return data

def validate_structure(data: dict) -> bool:
    return (
        isinstance(data, dict)
        and 'keywords' in data
        and 'entities' in data
        and 'context_passage' in data
    )

def fill_defaults(data: dict) -> dict:
    return {
        "keywords": data.get("keywords", []),
        "entities": data.get("entities", {"PERSON": [], "ORG": [], "DATE": [], "MISC": []}),
        "context_passage": data.get("context_passage", "")
    }
