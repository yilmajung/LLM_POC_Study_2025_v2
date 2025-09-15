import json, math, numpy as np, pandas as pd, torch
from typing import List, Tuple, Optional, Dict, Iterable
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import os
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel

# -----------------------------
# Constants
# -----------------------------
CANON_5 = ["strong_anti","anti","neutral","pro","strong_pro"]
CANON5_CODE = {k:i for i,k in enumerate(CANON_5)}
OPT_TOKENS = ["<OPT_STRONG_ANTI>","<OPT_ANTI>","<OPT_NEUTRAL>","<OPT_PRO>","<OPT_STRONG_PRO>"]
CANON2TOKEN = {
    "strong_anti": "<OPT_STRONG_ANTI>",
    "anti": "<OPT_ANTI>",
    "neutral": "<OPT_NEUTRAL>",
    "pro": "<OPT_PRO>",
    "strong_pro": "<OPT_STRONG_PRO>",
}

def dirichlet_row_smooth(counts_row, alpha=0.25):
    r = np.asarray(counts_row, dtype=float) + alpha
    s = r.sum()
    return (r / s) if s > 0 else np.ones_like(r)/len(r)

def make_prompt(survey, year_t, year_t1, group_meta: dict, question_text: str, from_canon: str):
    group_str = "; ".join([f"{k}={v}" for k,v in group_meta.items() if (v is not None and not (isinstance(v,float) and np.isnan(v)))])
    return (
        "[Task: Predict transition distribution]\n"
        f"Survey: {survey}\n"
        f"From wave: {year_t}  →  To wave: {year_t1}\n"
        f"Group: {group_str if group_str else 'all'}\n"
        f"Question: {question_text}\n"
        f"From option: {CANON2TOKEN[from_canon]}\n"
    )


OPT_TOKENS = ["<OPT_STRONG_ANTI>", "<OPT_ANTI>", "<OPT_NEUTRAL>", "<OPT_PRO>", "<OPT_STRONG_PRO>"]

import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

OPT_TOKENS = ["<OPT_STRONG_ANTI>","<OPT_ANTI>","<OPT_NEUTRAL>","<OPT_PRO>","<OPT_STRONG_PRO>"]

def load_tllm_for_eval_gpu(adapter_dir: str, base_id: str | None = None, dtype="bf16"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftConfig, PeftModel

    assert torch.cuda.is_available(), "No CUDA device visible."
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    OPT_TOKENS = ["<OPT_STRONG_ANTI>","<OPT_ANTI>","<OPT_NEUTRAL>","<OPT_PRO>","<OPT_STRONG_PRO>"]

    if base_id is None:
        cfg = PeftConfig.from_pretrained(adapter_dir)
        base_id = cfg.base_model_name_or_path

    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True) \
          if os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) \
          else AutoTokenizer.from_pretrained(base_id, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.add_special_tokens({"additional_special_tokens": [t for t in OPT_TOKENS if t not in tok.get_vocab()]})

    # Explicit device map to GPU 0; no offload
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        device_map={"": 0},          # put whole model on cuda:0
        low_cpu_mem_usage=False,
    )
    if base.get_input_embeddings().weight.shape[0] != len(tok):
        base.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        device_map={"": 0},          # keep adapters on cuda:0
        offload_folder=None
    )
    model.eval()
    return tok, model




@torch.no_grad()
def predict_transition_dist(model, tokenizer, prompt_text, opt_tokens, max_len=512):
    text = prompt_text + "Options: " + " ".join(opt_tokens) + "\nAnswer:\n"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    out = model(**enc)
    B = enc["input_ids"].shape[0]
    last_idx = enc["attention_mask"].sum(dim=1) - 1                   # [B]
    # gather per-sample logits at the last token -> [B, V]
    batch_idx = torch.arange(B)
    logits_last = out.logits[batch_idx, last_idx, :]                  # [B, V]
    # select option-token logits
    ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in opt_tokens], dtype=torch.long)
    option_logits = logits_last[:, ids]                                # [B, K]
    probs = torch.softmax(option_logits, dim=1).cpu().numpy()          # [B, K]
    return probs[0]  # if B==1


def preflight_prompt_check(tokenizer, model, prompt_text, opt_tokens=OPT_TOKENS):
    # Check all option tokens exist and are < vocab size
    ids = [tokenizer.convert_tokens_to_ids(t) for t in opt_tokens]
    vocab_size = model.get_input_embeddings().weight.shape[0]
    assert all(i is not None and i >= 0 for i in ids), f"Some option tokens not found: {list(zip(opt_tokens, ids))}"
    assert max(ids) < vocab_size, f"Option token id {max(ids)} exceeds model vocab {vocab_size}"

    # Tokenize once and ensure last_idx makes sense
    enc = tokenizer(prompt_text + "Options: " + " ".join(opt_tokens) + "\nAnswer:\n",
                    return_tensors="pt", truncation=True, max_length=512)
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1
    assert (last_idx >= 0).all(), "Negative last_idx from attention_mask (empty prompt?)"
    return True


# -----------------------------
# Build ground-truth rows from GSS panel
# -----------------------------
def build_gt_rows(
    df_long: pd.DataFrame,
    wave_pairs: List[Tuple[int,int]],
    group_cols_static: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
    survey_label: str = "GSS",
    question_text: str = "Harmonized attitude toward abortion (GSS collapsed)",
    smoothing_alpha: float = 0.25,
) -> List[Dict]:
    """
    Returns a list of row dicts:
      {
        'year_t', 'year_t1', 'group': {...}, 'from': <canon_label>,
        'to_dist_true': [K], 'n_from': float,
        'prompt_text': <prompt used for model inference>
      }
    """
    K = len(CANON_5)
    rows = []

    # prepare helper: code att5
    df = df_long.copy()
    df["att5_code"] = df["att5"].map(CANON5_CODE)

    # group handling
    grouped_all = [((), df)] if not group_cols_static else df.groupby(group_cols_static, dropna=False)

    for gkeys, gdf in grouped_all:
        group_meta = {}
        if group_cols_static:
            if not isinstance(gkeys, tuple):
                gkeys = (gkeys,)
            group_meta = {c: v for c, v in zip(group_cols_static, gkeys)}

        for (y_t, y_t1) in wave_pairs:
            # Left join to align same pid across t and t1
            df_t  = gdf[gdf["year"] == y_t][["yearid","att5","att5_code"] + ([weight_col] if weight_col else [])]
            df_t1 = gdf[gdf["year"] == y_t1][["yearid","att5","att5_code"] + ([weight_col] if weight_col else [])]
            merged = df_t.merge(df_t1, on="yearid", suffixes=("_t","_t1"))

            if merged.empty:
                continue

            # weight
            if weight_col:
                w = merged[f"{weight_col}_t"].fillna(0).to_numpy(float)
            else:
                w = np.ones(len(merged), dtype=float)

            # compute counts matrix KxK
            mat = np.zeros((K, K), dtype=float)
            for fi, ti, wi in zip(merged["att5_code_t"], merged["att5_code_t1"], w):
                if np.isnan(fi) or np.isnan(ti): 
                    continue
                mat[int(fi), int(ti)] += wi

            # build one row per "from" category
            for i, from_label in enumerate(CANON_5):
                counts_row = mat[i, :]
                n_from = counts_row.sum()
                if n_from == 0:
                    # still emit a smoothed tiny row so model sees the prompt, but mark low support
                    to_dist_true = dirichlet_row_smooth(counts_row, alpha=smoothing_alpha)
                else:
                    to_dist_true = dirichlet_row_smooth(counts_row, alpha=smoothing_alpha)

                prompt_text = make_prompt(
                    survey=survey_label,
                    year_t=str(y_t),
                    year_t1=str(y_t1),
                    group_meta=group_meta,
                    question_text=question_text,
                    from_canon=from_label
                )

                rows.append({
                    "year_t": str(y_t),
                    "year_t1": str(y_t1),
                    "group": group_meta,
                    "from": from_label,
                    "to_dist_true": to_dist_true.tolist(),
                    "n_from": float(n_from),
                    "prompt_text": prompt_text
                })
    return rows

# -----------------------------
# Metrics
# -----------------------------
def kl_div(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)))

def js_div(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = 0.5*(p/ (p.sum()+eps) + q/ (q.sum()+eps))
    return 0.5*kl_div(p,m,eps) + 0.5*kl_div(q,m,eps)

def rmse(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    return np.sqrt(np.mean((p - q)**2))

def switch_rate(vec, baseline_idx):
    """Share moving away from baseline_idx."""
    vec = np.asarray(vec, float)
    if vec.sum() == 0: return 0.0
    return 1.0 - (vec[baseline_idx] / vec.sum())

def direction_of_change_acc(from_idx, true_dist, pred_dist):
    """Compare E[to]-from_idx signs (down/neutral/up)."""
    idxs = np.arange(len(true_dist))
    true_mu = np.sum(idxs * true_dist)
    pred_mu = np.sum(idxs * pred_dist)
    true_dir = np.sign(true_mu - from_idx)
    pred_dir = np.sign(pred_mu - from_idx)
    return 1.0 if true_dir == pred_dir else 0.0

# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate_tllm_on_gss(
    model_dir: str,
    gss_df: pd.DataFrame,
    wave_pairs: List[Tuple[int,int]],
    group_cols_static: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
    report_csv_path: Optional[str] = "eval_rows_gss.csv",
):
    # 1) load model (base + LoRA adapter automatically)
    # from peft import AutoPeftModelForCausalLM  # no longer needed; we use the loader
    tok, mdl = load_tllm_for_eval_gpu(model_dir, base_id=None)  # pass base_id if needed


    # tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # if tok.pad_token is None:
    #     tok.pad_token = tok.eos_token
    # tok.padding_side = "left"

    # mdl = AutoPeftModelForCausalLM.from_pretrained(
    #     model_dir,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # 2) build GT rows
    rows = build_gt_rows(
        df_long=gss_df,
        wave_pairs=wave_pairs,
        group_cols_static=group_cols_static,
        weight_col=weight_col,
        survey_label="GSS",
        question_text="Harmonized abortion attitude (GSS collapsed)"
    )

    # 3) predict and score
    recs = []
    for r in rows:
        pred = predict_transition_dist(mdl, tok, r["prompt_text"], opt_tokens=OPT_TOKENS)
        true = np.array(r["to_dist_true"], float)
        from_idx = CANON5_CODE[r["from"]]

        rec = {
            "year_t": r["year_t"], "year_t1": r["year_t1"],
            "group": json.dumps(r["group"], ensure_ascii=False),
            "from": r["from"],
            "n_from": r["n_from"],
            "kl": kl_div(true, pred),
            "js": js_div(true, pred),
            "rmse": rmse(true, pred),
            "switch_true": switch_rate(true, from_idx),
            "switch_pred": switch_rate(pred, from_idx),
            "dir_acc": direction_of_change_acc(from_idx, true, pred),
        }
        recs.append(rec)

    df_eval = pd.DataFrame(recs)

    # 4) aggregate metrics (micro = weighted by n_from; macro = simple mean)
    def wavg(s, w): 
        s = np.asarray(s, float); w = np.asarray(w, float)
        w = np.where(np.isfinite(w), w, 0.0); s = np.where(np.isfinite(s), s, 0.0)
        return (s*w).sum() / (w.sum() + 1e-12)

    out = {
        "rows": len(df_eval),
        "micro_kl": wavg(df_eval["kl"], df_eval["n_from"]),
        "micro_js": wavg(df_eval["js"], df_eval["n_from"]),
        "micro_rmse": wavg(df_eval["rmse"], df_eval["n_from"]),
        "micro_dir_acc": wavg(df_eval["dir_acc"], df_eval["n_from"]),
        "micro_switch_rmse": wavg((df_eval["switch_true"]-df_eval["switch_pred"]).abs(), df_eval["n_from"]),
        "macro_kl": df_eval["kl"].mean(),
        "macro_js": df_eval["js"].mean(),
        "macro_rmse": df_eval["rmse"].mean(),
        "macro_dir_acc": df_eval["dir_acc"].mean(),
        "macro_switch_rmse": (df_eval["switch_true"]-df_eval["switch_pred"]).abs().mean(),
    }

    if report_csv_path:
        df_eval.to_csv(report_csv_path, index=False)

    return out, df_eval
