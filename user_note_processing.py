# user_note_processing.py

from pathlib import Path
import json
from typing import List, Dict

def load_user_notes(user_json_path: Path) -> List[Dict]:
    """
    Load manually written user notes from a JSON file.

    Args:
        user_json_path (Path): Path to the user-written JSON file.

    Returns:
        List[Dict]: A list of user memory entries, each entry as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the loaded file is not a valid list of dictionaries.
    """
    if not user_json_path.exists():
        raise FileNotFoundError(f"❌ User notes file not found: {user_json_path}")

    with open(user_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"❌ Expected a list of user notes, but got {type(data)}.")

    # 可以在这里额外加一层检查，比如检查每条数据是否有timestamp/description/source
    for idx, entry in enumerate(data):
        if not all(key in entry for key in ("timestamp", "description", "image_path", "source")):
            raise ValueError(f"❌ Invalid entry format at index {idx}: {entry}")

    print(f"✅ Loaded {len(data)} user-written notes from {user_json_path.name}")
    return data
