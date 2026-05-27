import json
import os
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

os.makedirs("results", exist_ok=True)

audio = MonoLoader(
    filename="input_16k_mono.wav",
    sampleRate=16000,
    resampleQuality=4,
)()

model = TensorflowPredictMusiCNN(
    graphFilename="models/msd-musicnn-1.pb",
    patchHopSize=187,  # non-overlapping patches to match essentia.js chunking
)
activations = model(audio)  # shape (N_patches, 50) — default output, not embedding

with open("models/msd-musicnn-1.json") as f:
    meta = json.load(f)
tags = meta["classes"]
assert len(tags) == 50, f"Expected 50 tags in metadata, got {len(tags)}"
assert activations.shape[1] == 50, (
    f"Expected model output dim 50, got {activations.shape[1]}. "
    f"Wrong output node? (default output should be tag sigmoids, not embedding.)"
)

mean_act = activations.mean(axis=0)
top10_idx = np.argsort(mean_act)[::-1][:10]

out = {
    "model": "msd-musicnn-1.pb",
    "audio": "input_16k_mono.wav",
    "n_patches": int(activations.shape[0]),
    "tags": tags,
    "mean_activations": mean_act.tolist(),
    "top10": [
        {"tag": tags[i], "score": float(mean_act[i])} for i in top10_idx
    ],
}

with open("results/python_output.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Wrote results/python_output.json (n_patches={out['n_patches']})")
for entry in out["top10"]:
    print(f"  {entry['tag']:>20s}: {entry['score']:.4f}")
