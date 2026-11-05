"""Display committed activation counts without loading the model."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from collect import DEFAULT_OUT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
manifest = json.loads((args.output / "manifest.json").read_text())
receipts = [json.loads(p.read_text()) for p in (args.output / "receipts").glob("*.json")]
c4_end = manifest["settings"]["c4_samples"] * len(manifest["labels"])
c4_messages = sum(max(0, min(r["message_end"], c4_end) - r["message_start"]) for r in receipts)
messages = sum(r["message_end"] - r["message_start"] for r in receipts)
print(json.dumps(dict(
    time=datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
    saved_shards=len(receipts), planned_shards=len(manifest["shards"]),
    saved_tokens=sum(r["tokens"] for r in receipts), planned_tokens=manifest["tokens"],
    saved_gib=round(sum(r["bytes"] for r in receipts) / 2**30, 2),
    saved_messages_by_source=dict(c4=c4_messages, dolma3=messages - c4_messages),
    verified_complete=(args.output / "complete.json").exists(),
    workers=[json.loads(p.read_text()) for p in sorted(args.output.glob("progress-worker-*.json"))],
), ensure_ascii=False))
