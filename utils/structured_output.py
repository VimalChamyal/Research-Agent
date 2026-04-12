import json
import re

def clean_json_string(text: str):
    """
    Clean common LLM JSON issues
    """
    # remove markdown
    text = text.strip().strip("```").strip()

    # remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


def extract_json(text: str):
    """
    Extract JSON from messy LLM output
    """

    # Step 1: try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Step 2: extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group()
        json_str = clean_json_string(json_str)

        try:
            return json.loads(json_str)
        except Exception as e:
            print("⚠️ JSON still invalid, trying fallback...")

    # Step 3: last fallback (VERY IMPORTANT)
    return {
        "evidence": []
    }


def parse_output(text, model):
    data = extract_json(text)

    try:
        return model.model_validate(data)
    except Exception as e:
        print("⚠️ Validation failed, returning empty structure")

        # fallback based on model
        if model.__name__ == "EvidencePack":
            return model(evidence=[])

        raise e