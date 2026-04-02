import os, re, sys, argparse
from typing import List
from unicodedata import normalize

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer

# ─── resolve sparse_coding on sys.path ────────────────────────────────────────
_SPARSE_CODING = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sparse_coding"))
if _SPARSE_CODING not in sys.path:
    sys.path.insert(0, _SPARSE_CODING)

from autoencoders.sae_ensemble import FunctionalTiedSAE
from activation_dataset import make_sentence_dataset_with_Nocot

K_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MAX_K  = max(K_LIST)

parser = argparse.ArgumentParser()
parser.add_argument("--model",          default="EleutherAI/pythia-70m-deduped")
parser.add_argument("--layer",          type=int, default=2)
parser.add_argument("--layer_loc",      default="resid")
parser.add_argument("--rank",           type=int, default=6,
                    help="SAE rank index 0-8; rank 6 ≈ l1_alpha=0.0027 (paper default)")
parser.add_argument("--dict_nocot",     required=True)
parser.add_argument("--dict_cot",       required=True)
parser.add_argument("--acts_nocot_dir", required=True)
parser.add_argument("--acts_cot_dir",   required=True)
parser.add_argument("--max_samples",    type=int, default=2000)
parser.add_argument("--out",            default=os.path.join(os.path.dirname(__file__), "results", "patch_curve.png"))
parser.add_argument("--save_stats",     default=os.path.join(os.path.dirname(__file__), "results", "ttest_results.txt"))
args = parser.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(args.out)),       exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.save_stats)), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)


def build_gsm8k_lookup(split="train"):
    ds = load_dataset("openai/gsm8k", split=split, verification_mode="no_checks")
    table = {}
    for ex in ds:
        q = normalize("NFKC", ex["question"].strip())
        a = ex["answer"].split("####")[-1].strip()
        table[q] = a
    return table


GSM_LOOKUP = build_gsm8k_lookup()
print("GSM lookup size =", len(GSM_LOOKUP))

model: HookedTransformer = HookedTransformer.from_pretrained(
    args.model,
    device=device,
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
)
tokenizer = model.tokenizer
print(f"Loaded {args.model}")


def load_dict(p):
    return torch.load(p, map_location="cpu")


dict_nc = load_dict(args.dict_nocot)
dict_c  = load_dict(args.dict_cot)
sae_nc  = dict_nc[args.layer][args.rank][0]
sae_nc.to_device(device)
sae_c = dict_c[args.layer][args.rank][0]
sae_c.to_device(device)
print("SAE loaded. dict size =", sae_nc.n_feats)


def extract_question_from_prompt(txt: str) -> str:
    try:
        q = txt.split("Q:")[1].split("\nA:")[0]
    except Exception:
        q = txt
    return normalize("NFKC", q.strip())


def answer_logprob(model, prompt_ids, answer, hooks=None):
    ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    ids = torch.cat(
        [prompt_ids.squeeze(0), torch.tensor(ans_ids, device=prompt_ids.device)], 0
    )
    with model.hooks(hooks or []):
        logits = model(ids[:-1].unsqueeze(0), return_type="logits")
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt  = ids[1:].unsqueeze(0)
    return logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum().item()


def sorted_pt_files(
    folder: str, *, numeric_only=True, prefix: str = "", layer_tag: str = ""
) -> List[str]:
    paths = []
    for f in os.listdir(folder):
        if not f.endswith(".pt"):
            continue
        if prefix and not f.startswith(prefix):
            continue
        if layer_tag and layer_tag not in f:
            continue
        if numeric_only and not re.fullmatch(r"\d+\.pt", f):
            continue
        paths.append(os.path.join(folder, f))
    paths.sort(key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    return paths


def run_direction(acts_from_dir, acts_to_dir, sae_from, sae_to, tag):
    files_f = sorted_pt_files(acts_from_dir, numeric_only=True)
    files_t = sorted_pt_files(acts_to_dir,   numeric_only=True)
    files_i = sorted_pt_files(
        acts_from_dir, numeric_only=False, prefix="input_ids_layer", layer_tag="chunk"
    )
    assert len(files_f) == len(files_t) == len(files_i)
    results = {k: [] for k in K_LIST}
    smp  = 0
    pbar = tqdm(total=args.max_samples, desc=tag)
    for f_f, f_t, f_i in zip(files_f, files_t, files_i):
        if smp >= args.max_samples:
            break
        acts_f  = torch.load(f_f, map_location=device).to(torch.bfloat16)
        acts_t  = torch.load(f_t, map_location=device).to(torch.bfloat16)
        ids_ls  = torch.load(f_i)
        idx_start = 0
        for ids in ids_ls:
            if smp >= args.max_samples:
                break
            idx_end = idx_start + ids.shape[0] - 1
            a_f = acts_f[idx_end : idx_end + 1].to(device, sae_from.encoder.dtype)
            a_t = acts_t[idx_end : idx_end + 1].to(device, sae_to.encoder.dtype)
            prompt_ids = ids.unsqueeze(0).to(device)

            txt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            q   = extract_question_from_prompt(txt)
            ans = GSM_LOOKUP.get(q, None)
            if ans is None:
                idx_start = idx_end + 1
                continue

            base_ll = answer_logprob(model, prompt_ids, ans)
            c_f  = sae_from.encode(a_f)
            c_t  = sae_to.encode(a_t)
            diff = (c_t - c_f).abs().squeeze(0)

            for K in K_LIST:
                topk = diff.topk(k=K).indices
                code = c_f.clone()
                code[0, topk] = c_t[0, topk]
                patched = sae_from.decode(code).squeeze(0)

                def hk(resid, hook):
                    resid[:, -1, :] = patched.to(resid.device)
                    return resid

                hooks      = [(f"blocks.{args.layer}.hook_{args.layer_loc}_pre", hk)]
                patched_ll = answer_logprob(model, prompt_ids, ans, hooks)
                results[K].append(patched_ll - base_ll)

            smp += 1
            pbar.update(1)
            idx_start = idx_end + 1
        del acts_f, acts_t
        torch.cuda.empty_cache()
    pbar.close()
    return results


def run_direction_random(acts_from_dir, acts_to_dir, sae_from, sae_to, tag):
    files_f = sorted_pt_files(acts_from_dir, numeric_only=True)
    files_t = sorted_pt_files(acts_to_dir,   numeric_only=True)
    files_i = sorted_pt_files(
        acts_from_dir, numeric_only=False, prefix="input_ids_layer", layer_tag="chunk"
    )
    assert len(files_f) == len(files_t) == len(files_i)
    results = {k: [] for k in K_LIST}
    smp  = 0
    pbar = tqdm(total=args.max_samples, desc=f"{tag}-Random")
    for f_f, f_t, f_i in zip(files_f, files_t, files_i):
        if smp >= args.max_samples:
            break
        acts_f  = torch.load(f_f, map_location=device).to(torch.bfloat16)
        acts_t  = torch.load(f_t, map_location=device).to(torch.bfloat16)
        ids_ls  = torch.load(f_i)
        idx_start = 0
        for ids in ids_ls:
            if smp >= args.max_samples:
                break
            idx_end = idx_start + ids.shape[0] - 1
            a_f = acts_f[idx_end : idx_end + 1].to(device, sae_from.encoder.dtype)
            a_t = acts_t[idx_end : idx_end + 1].to(device, sae_to.encoder.dtype)
            prompt_ids = ids.unsqueeze(0).to(device)

            txt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            q   = extract_question_from_prompt(txt)
            ans = GSM_LOOKUP.get(q, None)
            if ans is None:
                idx_start = idx_end + 1
                continue

            base_ll = answer_logprob(model, prompt_ids, ans)
            c_f  = sae_from.encode(a_f)
            c_t  = sae_to.encode(a_t)

            for K in K_LIST:
                rand_idx = torch.randperm(c_f.shape[-1])[:K]
                code = c_f.clone()
                code[0, rand_idx] = c_t[0, rand_idx]
                patched = sae_from.decode(code).squeeze(0)

                def hk(resid, hook):
                    resid[:, -1, :] = patched.to(resid.device)
                    return resid

                hooks      = [(f"blocks.{args.layer}.hook_{args.layer_loc}_pre", hk)]
                patched_ll = answer_logprob(model, prompt_ids, ans, hooks)
                results[K].append(patched_ll - base_ll)

            smp += 1
            pbar.update(1)
            idx_start = idx_end + 1
        del acts_f, acts_t
        torch.cuda.empty_cache()
    pbar.close()
    return results


print(">>> CoT → NoCoT  (Top-K)")
res_c2n = run_direction(args.acts_cot_dir, args.acts_nocot_dir, sae_c, sae_nc, "CoT→NoCoT")
print(">>> NoCoT → CoT  (Top-K)")
res_n2c = run_direction(args.acts_nocot_dir, args.acts_cot_dir, sae_nc, sae_c, "NoCoT→CoT")
print(">>> CoT → NoCoT  (Random-K)")
res_c2n_random = run_direction_random(args.acts_cot_dir, args.acts_nocot_dir, sae_c, sae_nc, "CoT→NoCoT")
print(">>> NoCoT → CoT  (Random-K)")
res_n2c_random = run_direction_random(args.acts_nocot_dir, args.acts_cot_dir, sae_nc, sae_c, "NoCoT→CoT")


# ── Statistical analysis ──────────────────────────────────────────────────────
def perform_statistical_analysis(res_c2n, res_n2c, save_path):
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Statistical Analysis Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. One-sample t-tests (H0: mean = 0)\n")
        f.write("-" * 40 + "\n")
        for direction, results in [("CoT → NoCoT", res_c2n), ("NoCoT → CoT", res_n2c)]:
            f.write(f"\n{direction}:\n")
            for k in K_LIST:
                data = np.array(results[k])
                if len(data) > 1:
                    t, p = stats.ttest_1samp(data, 0)
                    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    f.write(f"  K={k:3d}: mean={data.mean():+.4f}, std={data.std(ddof=1):.4f}, "
                            f"t={t:+.3f}, p={p:.6f} {sig}\n")

        f.write("\n\n2. Paired t-tests (CoT→NoCoT vs NoCoT→CoT)\n")
        f.write("-" * 50 + "\n")
        for k in K_LIST:
            d1 = np.array(res_c2n[k])
            d2 = np.array(res_n2c[k])
            n  = min(len(d1), len(d2))
            if n > 1:
                t, p   = stats.ttest_rel(d1[:n], d2[:n])
                diff_m = (d1[:n] - d2[:n]).mean()
                sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                f.write(f"K={k:3d}: diff_mean={diff_m:+.4f}, t={t:+.3f}, p={p:.6f} {sig}\n")

        f.write("\n\n3. Summary Statistics\n")
        f.write("-" * 30 + "\n")
        for direction, results in [("CoT → NoCoT", res_c2n), ("NoCoT → CoT", res_n2c)]:
            f.write(f"\n{direction}:\n")
            pos_ks, neg_ks = [], []
            for k in K_LIST:
                data = np.array(results[k])
                if len(data):
                    pos = np.sum(data > 0)
                    f.write(f"  K={k:3d}: {pos:4d}/{len(data):4d} positive "
                            f"({pos/len(data)*100:5.1f}%), mean={data.mean():+.4f}\n")
                    (pos_ks if data.mean() > 0 else neg_ks).append(k)
            f.write(f"  Positive effect at K: {pos_ks}\n")
            f.write(f"  Negative effect at K: {neg_ks}\n")

        f.write(f"\nSignificance: * p<0.05, ** p<0.01, *** p<0.001\n")
        f.write(f"Samples per direction: ~{len(res_c2n[K_LIST[0]])}\n")


perform_statistical_analysis(res_c2n, res_n2c, args.save_stats)
print(f"Statistical analysis saved → {args.save_stats}")


# ── Patch-curve plot ──────────────────────────────────────────────────────────
def mean_ci(vals):
    arr = np.array(vals)
    m   = arr.mean()
    se  = arr.std(ddof=1) / np.sqrt(len(arr))
    return m, 1.96 * se


def build_series(res_dict):
    ks = sorted(res_dict)
    means, cis = zip(*(mean_ci(res_dict[k]) for k in ks))
    return np.array(ks), np.array(means), np.array(cis)


k1, m1, c1 = build_series(res_c2n)
k2, m2, c2 = build_series(res_n2c)
k3, m3, c3 = build_series(res_c2n_random)
k4, m4, c4 = build_series(res_n2c_random)

plt.figure(figsize=(6, 4))
plt.plot(k1, m1, label="CoT→NoCoT (Top-K)",    color="tab:orange")
plt.fill_between(k1, m1 - c1, m1 + c1, alpha=0.2, color="tab:orange")
plt.plot(k2, m2, label="NoCoT→CoT (Top-K)",    color="tab:blue")
plt.fill_between(k2, m2 - c2, m2 + c2, alpha=0.2, color="tab:blue")
plt.plot(k3, m3, label="CoT→NoCoT (Random-K)", color="tab:gray",  linestyle="--")
plt.fill_between(k3, m3 - c3, m3 + c3, alpha=0.1, color="tab:gray")
plt.plot(k4, m4, label="NoCoT→CoT (Random-K)", color="tab:green", linestyle="--")
plt.fill_between(k4, m4 - c4, m4 + c4, alpha=0.1, color="tab:green")

plt.axhline(0, ls="--", lw=0.8, c="k")
plt.xscale("log", base=2)
plt.xlabel("K (patched features)")
plt.ylabel("Δ log-prob")
plt.title(f"{args.model}  Top-K vs Random-K Patch Curve")
plt.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Plot saved → {args.out}")


# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY STATISTICAL RESULTS")
print("=" * 50)
for direction, results in [("CoT → NoCoT", res_c2n), ("NoCoT → CoT", res_n2c)]:
    print(f"\n{direction}:")
    sig_ks = []
    for k in K_LIST:
        data = np.array(results[k])
        if len(data) > 1:
            t, p = stats.ttest_1samp(data, 0)
            if p < 0.05:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                print(f"  K={k}: mean={data.mean():+.4f}, p={p:.4f} {sig}")
                sig_ks.append(k)
    if not sig_ks:
        print("  No significant effects found")

print(f"\nDetailed results → {args.save_stats}")
print("=" * 50)
