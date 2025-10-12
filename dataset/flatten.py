import json
from copy import deepcopy
from typing import Any, Dict, List

# Configure which extra_info keys are known to be nested and how to flatten them.
# For each key in extra_info that may contain a dict, we define:
# - prefix: the prefix to use when lifting nested fields into extra_info
# - strategy: "flatten" tries to lift simple scalars and JSON-stringify complex subfields
#             "stringify" JSON-stringifies the entire nested object into one field
FLATTEN_SCHEMA = {
    "input_bbox": {"prefix": "bbox_", "strategy": "flatten"},
    "input_point": {"prefix": "point_", "strategy": "flatten"},
    "input_mask": {"prefix": "mask_", "strategy": "flatten"},
    # add other known nested keys here if needed
}

# Helper: determine if a value is a "simple" scalar that can live in extra_info directly
def is_simple_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None

# Flatten a nested dict into parent with a given prefix.
# - Simple scalars become extra_info[prefix+key] = value
# - Lists or dicts become extra_info[prefix+key] = json.dumps(value, ensure_ascii=False)
def flatten_dict_into(extra_info: Dict[str, Any], nested: Dict[str, Any], prefix: str) -> None:
    for k, v in nested.items():
        flat_key = f"{prefix}{k}"
        if is_simple_scalar(v):
            extra_info[flat_key] = v
        else:
            # For lists/dicts/other, store as JSON string
            try:
                extra_info[flat_key] = json.dumps(v, ensure_ascii=False)
            except Exception:
                # Fallback to str if serialization fails
                extra_info[flat_key] = str(v)

# Main transform: remove nested dicts from extra_info and replace with flattened fields
def transform_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = deepcopy(sample)
    extra = sample.get("extra_info")
    if not isinstance(extra, dict):
        return sample

    # Iterate over keys that might contain nested dicts
    for nested_key, config in FLATTEN_SCHEMA.items():
        if nested_key in extra and isinstance(extra[nested_key], dict):
            nested_obj = extra[nested_key]
            strategy = config.get("strategy", "flatten")
            prefix = config.get("prefix", f"{nested_key}_")

            if strategy == "flatten":
                flatten_dict_into(extra, nested_obj, prefix)
            elif strategy == "stringify":
                # Store whole object into one flat field
                extra[prefix.rstrip("_")] = json.dumps(nested_obj, ensure_ascii=False)

            # Remove the original nested dict to ensure no dicts remain inside extra_info
            del extra[nested_key]

    # Also ensure any remaining values in extra_info are scalars; stringify others
    for k, v in list(extra.items()):
        if isinstance(v, (dict, list)):
            try:
                extra[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                extra[k] = str(v)

    sample["extra_info"] = extra
    return sample

def transform_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_sample(s) for s in data]

if __name__ == "__main__":
    # Example: read from input.json and write to output.json
    import argparse, sys, pathlib

    parser = argparse.ArgumentParser(description="Flatten nested dicts inside extra_info.")
    parser.add_argument("-i", "--input", type=pathlib.Path, required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True, help="Path to output JSON file")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Input JSON must be a list of samples.", file=sys.stderr)
        sys.exit(1)

    out = transform_dataset(data)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Transformed {len(out)} samples -> {args.output}")