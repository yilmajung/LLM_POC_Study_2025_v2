import json, argparse, math, random
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable


CANON = ["strong_anti", "anti", "neutral", "pro", "strong_pro"]  

def format_group_meta(group: Dict) -> str:
    """
    Render group dict into a compact, stable string.
    We omit None values; cast everything else to str.
    """
    items = []
    for k in sorted(group.keys()):
        v = group[k]
        if v is None: 
            continue
        items.append(f"{k}={v}")
    return "; ".join(items) if items else "all"

def build_prompt_text(rec: Dict, from_label: str) -> str:
    """
    Construct the prompt (without the Options: line).
    Your Trainer adds the Options: and Answer: lines and handles order/shuffle.
    """
    survey = rec.get("survey_id", "UAS")
    year_t = rec["year_t"]
    year_t1 = rec["year_t1"]
    q_text = rec.get("question_text", "Attitudes toward abortion over the years")
    group_str = format_group_meta(rec.get("group", {}))
    
    prompt = (
        "[Task: Predict transition distribution]\n"
        f"Survey: {survey}\n"
        f"From year: {year_t}  →  To year: {year_t1}\n"
        f"Group: {group_str}\n"
        f"Question: {q_text}\n"
        f"From option: {from_label}\n"
    )
    return prompt

def expand_record_to_rows(rec: Dict, use_log_weight: bool = False, min_weight: float = 1.0) -> List[Dict]:
    """
    Turn one transition record into |CANON| row-level examples.
    """
    rows = rec["transition_rows"]  # list like [{"from": "anti", "to_dist": [...]}, ...]
    n_from = rec.get("n_from", {}) # dict like {"anti": 153.7, ...}
    out = []

    # Build a fast map from 'from' => to_dist
    row_map = {r["from"]: r["to_dist"] for r in rows}

    for from_label in CANON:
        if from_label not in row_map:
            # If this state wasn't present in this record, skip (or create uniform tiny target)
            continue

        to_dist = row_map[from_label]
        weight_raw = float(n_from.get(from_label, 0.0))
        # guard against zero/neg:
        weight_raw = max(weight_raw, 0.0)

        # Optional: log-scale to avoid huge gradients; keep a minimum positive weight.
        if use_log_weight:
            w = math.log1p(weight_raw)
        else:
            w = weight_raw
        w = max(w, min_weight)

        prompt_text = build_prompt_text(rec, from_label)
        out.append({
            "prompt_text": prompt_text,
            "to_dist": to_dist,
            "weight": w
        })
    return out

def latest_target_waves(records: List[Dict]) -> List[str]:
    """
    Return a list of 'year_t1' values sorted, and the max (latest) for time-based split.
    """
    t1s = sorted({r["year_t1"] for r in records})
    return t1s

def time_based_split(rows: List[Tuple[str, Dict]], latest_t1: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by whether the parent record's year_t1 equals the latest target wave.
    rows is a list of (year_t1, row_dict).
    """
    train, val = [], []
    for t1, row in rows:
        (val if t1 == latest_t1 else train).append(row)
    return train, val

def random_split(rows: List[Dict], val_ratio: float = 0.1, seed: int = 123) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    r = rows[:]
    random.shuffle(r)
    n_val = max(1, int(len(r) * val_ratio))
    return r[n_val:], r[:n_val]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="TLLM_abortion_transitions_tvEdu.jsonl")
    ap.add_argument("--train_out", default="train_rows.jsonl")
    ap.add_argument("--val_out", default="val_rows.jsonl")
    ap.add_argument("--split", choices=["time","random"], default="time",
                    help="time: holdout latest year_t1; random: random 10%% val")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="used only when --split random")
    ap.add_argument("--log_weight", action="store_true", help="use log(1+n_from) as weight")
    ap.add_argument("--min_weight", type=float, default=1.0)
    args = ap.parse_args()

    # 1) Read pair-level records
    recs = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            recs.append(json.loads(line))

    # 2) Expand to row-level
    row_list: List[Tuple[str, Dict]] = []
    for rec in recs:
        year_t1 = rec["year_t1"]
        rows = expand_record_to_rows(rec, use_log_weight=args.log_weight, min_weight=args.min_weight)
        for r in rows:
            row_list.append((year_t1, r))

    # 3) Split into train / val
    if args.split == "time":
        t1s = latest_target_waves(recs)
        latest_t1 = t1s[-1]  # hold out latest target wave for validation
        train_rows, val_rows = time_based_split(row_list, latest_t1)
        # ---- Normalize to dicts for writing ----
        train_rows = [r if isinstance(r, dict) else r[1] for r in train_rows]
        val_rows   = [r if isinstance(r, dict) else r[1] for r in val_rows]
    else:
        # random split returns dicts already
        _, only_rows = zip(*row_list) if row_list else ([], [])
        train_rows, val_rows = random_split(list(only_rows), val_ratio=args.val_ratio)

    # 4) Write outputs
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    with open(args.val_out, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows.")

if __name__ == "__main__":
    main()
