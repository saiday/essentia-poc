# MSD-MusiCNN Cross-Implementation Parity Demo

對同一份音檔分別用 Python (`essentia.standard.TensorflowPredictMusiCNN`) 與 JavaScript (`essentia.js` + tfjs CPU backend) 跑 MSD-MusiCNN auto-tagging,比對兩邊 50 維 sigmoid 輸出是否一致。

## How to run

1. 一次性設定 (建 venv、安裝 deps、下載模型):

   ```
   bash setup.sh
   ```

2. 放一份音檔在 `./input.mp3` (任何 ffmpeg 能解的格式都行)。

3. 跑測試:

   ```
   bash run_test.sh ./input.mp3
   ```

## What to expect

`run_test.sh` 跑完會產生:

- `input_16k_mono.wav` — 兩端共用的 16 kHz 單聲道 float32 PCM
- `results/python_output.json` — Python 端 50 維 mean activation + top-10 tags
- `results/js_output.json` — JS 端對應輸出 (相同格式)
- `results/comparison_report.txt` — shape 對齊、逐 tag diff 統計、Jaccard@5/10、Spearman ρ、PASS/FAIL 判定

判定門檻:

- 嚴格 PASS: `max |diff| < 0.05` **且** Top-10 Jaccard ≥ 0.8
- 寬鬆 PASS: Top-10 Jaccard ≥ 0.6 且 Spearman ρ ≥ 0.9
- 都不到就 FAIL

## Known sources of difference

- **Mel-spectrogram backend**: Python 端由 essentia 的 C++ 實作算 mel;JS 端由 essentia.js 編譯出的 WASM 算。同樣的演算法、不同 toolchain,浮點數會有微小差異。
- **Inference backend**: Python 走 libtensorflow (C++);JS 強制走 `@tensorflow/tfjs` 的 pure-JS CPU backend (`tf.setBackend("cpu")`),啟動時硬性 assert 切換成功。我們刻意不裝 `@tensorflow/tfjs-node`,避免 native backend 偷偷接管 (那會讓兩端數值「太接近」而失去比對意義)。
- 因此即使權重相同,小幅數值偏差是預期的;判定門檻就是為了容許這個。

## Files

| File | Purpose |
|---|---|
| `setup.sh` | 建 venv、`pip install`、`npm install`、下載三份模型檔 |
| `preprocess.sh` | ffmpeg 把任意輸入 → 16 kHz mono float32 wav |
| `python_tagger.py` | essentia + TF C++ 推論,寫 `results/python_output.json` |
| `js_tagger.js` | essentia.js + tfjs CPU 推論,寫 `results/js_output.json` |
| `compare.py` | 比對兩邊輸出,寫 `results/comparison_report.txt` |
| `run_test.sh` | preprocess → python → js → compare 一鍵跑 |
