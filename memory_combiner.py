# memory_combiner.py

from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict

def combine_memories(user_data: List[Dict], model_data: List[Dict], output_path: Path) -> None:
    """
    Combine user-written and model-generated memory entries, sorted by timestamp, and save to JSON.

    Args:
        user_data (List[Dict]): List of user memory entries.
        model_data (List[Dict]): List of model-generated memory entries.
        output_path (Path): Output JSON file path.
    """
    combined = user_data + model_data

    # 按照 timestamp 排序
    combined.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M"))

    # 保存
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"✅ Combined {len(user_data)} user notes + {len(model_data)} model descriptions.")
    print(f"✅ Saved combined memories to {output_path.name}.")
