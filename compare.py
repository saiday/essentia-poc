import json
import os
import numpy as np
from scipy.stats import spearmanr

with open("results/python_output.json") as f:
    py = json.load(f)
with open("results/js_output.json") as f:
    js = json.load(f)

lines = []
lines.append("MSD-MusiCNN cross-implementation parity report")
lines.append("=" * 60)
lines.append("")
lines.append(f"Python: model={py['model']}, audio={py['audio']}")
lines.append(f"JS:     model={js['model']}, audio={js['audio']}")
lines.append("")

# ---- [1] Shape & tag alignment ----
lines.append("[1] Shape & tag alignment")
if py["tags"] != js["tags"]:
    lines.append("  FAIL: tag lists differ between Python and JS!")
    for i, (pt, jt) in enumerate(zip(py["tags"], js["tags"])):
        if pt != jt:
            lines.append(f"  first mismatch at index {i}: py={pt!r}, js={jt!r}")
            break
    report = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_report.txt", "w") as f:
        f.write(report + "\n")
    print(report)
    raise SystemExit(1)
lines.append(f"  Both sides have {len(py['tags'])} tags; lists identical.")
lines.append(f"  Python n_patches: {py['n_patches']}")
lines.append(f"  JS     n_patches: {js['n_patches']}")
warnings = []
if py["n_patches"] != js["n_patches"]:
    lines.append("  WARNING: n_patches differ — preprocessing not aligned.")
    warnings.append("n_patches mismatch")

# ---- [2] Per-tag probability differences ----
lines.append("")
lines.append("[2] Per-tag probability differences (python - js)")
a = np.asarray(py["mean_activations"], dtype=np.float64)
b = np.asarray(js["mean_activations"], dtype=np.float64)
diff = a - b
absdiff = np.abs(diff)
max_abs = float(absdiff.max())
mean_abs = float(absdiff.mean())
std_diff = float(diff.std())
lines.append(f"  max |diff|  = {max_abs:.6f}")
lines.append(f"  mean |diff| = {mean_abs:.6f}")
lines.append(f"  std diff    = {std_diff:.6f}")
top5_idx = np.argsort(absdiff)[::-1][:5]
lines.append("  Top 5 |diff| tags:")
for i in top5_idx:
    tag = py["tags"][i]
    lines.append(
        f"    {tag:>20s}:  py={a[i]:.4f}   js={b[i]:.4f}   diff={diff[i]:+.4f}"
    )

# ---- [3] Top-K comparison ----
lines.append("")
lines.append("[3] Top-K set comparison")

def top_k_tags(d, k):
    return set(t["tag"] for t in d["top10"][:k])

def jaccard(s1, s2):
    union = s1 | s2
    return (len(s1 & s2) / len(union)) if union else 1.0

top5_py = top_k_tags(py, 5)
top5_js = top_k_tags(js, 5)
top10_py = top_k_tags(py, 10)
top10_js = top_k_tags(js, 10)
jacc5 = jaccard(top5_py, top5_js)
jacc10 = jaccard(top10_py, top10_js)
rho, _ = spearmanr(a, b)

lines.append(f"  Jaccard@5  = {jacc5:.4f}")
lines.append(f"    py top5: {sorted(top5_py)}")
lines.append(f"    js top5: {sorted(top5_js)}")
lines.append(f"  Jaccard@10 = {jacc10:.4f}")
lines.append(f"    py top10: {sorted(top10_py)}")
lines.append(f"    js top10: {sorted(top10_js)}")
lines.append(f"  Spearman rho (50-dim) = {rho:.4f}")

# ---- [4] Verdict ----
lines.append("")
lines.append("[4] Verdict")
strict = (max_abs < 0.05) and (jacc10 >= 0.8)
lenient = (jacc10 >= 0.6) and (rho >= 0.9)
if strict:
    lines.append("  PASS (strict): max |diff| < 0.05 AND Jaccard@10 >= 0.8")
elif lenient:
    lines.append("  PASS (lenient): Jaccard@10 >= 0.6 AND Spearman rho >= 0.9")
    lines.append("  Note: did not meet strict criterion (max |diff| or Jaccard@10).")
else:
    lines.append("  FAIL: neither strict nor lenient criterion met.")
    lines.append("  Possible causes:")
    lines.append("    - mel-spectrogram mismatch (essentia C++ vs essentia.js WASM)")
    lines.append("    - tfjs backend not actually CPU (would shift numerics)")
    lines.append("    - different model version on each side")
    lines.append("    - preprocessing diverged (n_patches mismatch — see warnings)")

if warnings:
    lines.append("")
    lines.append("Warnings raised during comparison:")
    for w in warnings:
        lines.append(f"  - {w}")

report = "\n".join(lines)
os.makedirs("results", exist_ok=True)
with open("results/comparison_report.txt", "w") as f:
    f.write(report + "\n")
print(report)
