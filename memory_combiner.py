# memory_combiner.py

from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict

def combine_memories(user_data: List[Dict], model_data: List[Dict], output_path: Path) -> None:
    """
    Combine user-written and model-generated memory entries, sorted by timestamp, and save to JSON.
    """

    def parse_timestamp(ts: str) -> datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unknown timestamp format: {ts}")

    combined = user_data + model_data

    combined.sort(key=lambda x: parse_timestamp(x["timestamp"]))

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"✅ Combined {len(user_data)} user notes + {len(model_data)} model descriptions.")
    print(f"✅ Saved combined memories to {output_path.name}.")
