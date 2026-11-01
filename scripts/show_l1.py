"""Print what the L1 agent did on one chip, from its stored transcript. No model, no key, no network.

usage: python -I scripts/show_l1.py out/rescue/MAT_73_12_28_0020_LHZ
"""
import json
import sys
from pathlib import Path


def short(value, keep=230):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= keep else text[:keep] + " ..."


def main(folder):
    folder = Path(folder)
    if not (folder / "l1" / "l1_transcript.json").exists():
        sys.exit(f"{folder}: no l1/l1_transcript.json. Run rescue.py --l1 first, or use examples/MAT_73_12_28_0020_LHZ")
    log = json.loads((folder / "l1" / "l1_transcript.json").read_text())["log"]
    rec = json.loads((folder / "l1" / "record.json").read_text())
    l1 = rec.get("l1") or {}
    print(f"Muse Spark on {rec.get('chip')}: {l1.get('requests')} requests, {l1.get('tool_calls')} tool calls, ${l1.get('cost_usd')}, {l1.get('seconds')} s\n")
    for entry in log:
        if "request" in entry:
            print(f"request {entry['request']} ({entry.get('secs')} s)")
        elif "tool" in entry:
            args = ", ".join(f"{k}={v!r}" for k, v in (entry.get("args") or {}).items())
            print(f"  -> {entry['tool']}({args})")
            print(f"     {short(entry.get('result'))}\n")
        elif "final" in entry:
            print(f"  final answer: {short(entry['final'], 400)}")
            print(f"  validation errors: {entry.get('validation_errors') or 'none'}\n")
    ev = rec.get("main_event")
    print(f"record: {(rec.get('station') or {}).get('code')} {(rec.get('component') or {}).get('name')} ON {rec.get('on_utc')}")
    print(f"main event: M{ev.get('mag')} {ev.get('place')}, predicted P on line {ev.get('P_line')}" if ev else "main event: none picked")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
