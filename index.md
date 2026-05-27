---
layout: default
title: MSD-MusiCNN Python / JS 一致性 PoC — 交接文件
description: Python 與 JS 兩端 inference 一致性驗證 PoC 的交接說明
---

## 這個 PoC 在做什麼

同一份音檔，分別用兩種方式跑 **MSD-MusiCNN** auto-tagging 模型，然後比對兩邊的輸出是不是一致：

- **Python 端**：`essentia.standard.TensorflowPredictMusiCNN`（底層是 TensorFlow C++）
- **JS 端**：`essentia.js`(WASM)+ `@tensorflow/tfjs` 純 JS CPU backend

模型會輸出 50 維的 sigmoid 分數（對應 50 個 music tags）。我們比對兩端的 mean activation、Top-K tags、相關係數，判定是否一致(PASS / FAIL)。

目的：確認「同一個模型在 Python 跟瀏覽器 / Node 環境跑出來的結果可以對得起來」，作為之後把 inference 搬到前端的依據。

---

## 怎麼跑

```bash
# 1. 一次性設定(建 venv、裝套件、下載模型)
bash setup.sh

# 2. 放一份音檔在 ./input.mp3(任何 ffmpeg 能解的格式都行)

# 3. 一鍵跑測試
bash run_test.sh ./input.mp3
```

`run_test.sh` 的流程：`preprocess`（ffmpeg 轉 16kHz mono wav）→ Python inference → JS inference → 比對 → 印報告。

產出檔案在 `results/`：

| 檔案 | 內容 |
|---|---|
| `python_output.json` | Python 端 50 維 mean activation + Top-10 tags |
| `js_output.json` | JS 端對應輸出（相同格式） |
| `comparison_report.txt` | 逐 tag 差異統計、Jaccard、Spearman、PASS/FAIL 判定 |

---

## 我們改了什麼（重點）

這個 PoC 一開始兩端跑出來的 patch 數量對不上（Python 158、JS 80），代表兩邊其實不是在做同一件事。我們改了兩個地方，讓兩端真正對齊。**最後結果：patch 數量 79 = 79，Top-10 完全一致，strict PASS，沒有任何 warning。**

### 修改 1：Python 端改成「不重疊」切 patch — `python_tagger.py`

MusiCNN 把 mel-spectrogram 切成一段段 187 frames 的 patch 餵給模型。重點在「每個 patch 往前跳幾個 frame」：

- essentia(Python)預設 `patchHopSize=93`，等於 **每個 patch 重疊一半**，所以 patch 數量大約是兩倍。
- essentia.js(JS)只支援 **不重疊**切法（每次跳一整個 187），沒有重疊的選項。

兩邊預設不同，patch 數量自然對不上(158 vs 80)。因為 JS 端沒辦法改成重疊，所以改 Python 端配合 JS：

```python
model = TensorflowPredictMusiCNN(
    graphFilename="models/msd-musicnn-1.pb",
    patchHopSize=187,  # 預設是 93(重疊一半),改成 187 變成不重疊,對齊 JS
)
```

改完 Python 從 158 → 79 patch。

### 修改 2：JS 端丟掉尾巴不足一個 patch 的部分 — `js_tagger.js`

改完修改 1 之後還差一個：Python 79、JS 80。原因是「最後那段不滿一個 patch 的尾巴」兩邊處理方式不同：

- Python `lastPatchMode` 預設 `"discard"`：直接丟掉尾巴。
- JS 原本用 `zeroPadding=true`：把尾巴補 0 湊成一個完整 patch，所以多一個。

> 注意：essentia.js 的 `predict(features, false)` **不是**「丟尾巴」的意思。它會 assert 輸入剛好等於一個 patch(187 frames)，只能吃單一 patch，多 patch 會直接報錯。所以不能靠這個參數丟尾巴。

正確做法：在 predict 之前，先把 mel frames 截成 187 的整數倍（把尾巴切掉），再用 `zeroPadding=true`。剛好整數倍時 essentia.js 不會真的補 0，就會得到乾淨的 79 個 patch：

```javascript
const patch = features.patchSize;
const usableFrames = Math.floor(features.frameSize / patch) * patch; // 14872 → 14773 = 79 × 187
features.melSpectrum = features.melSpectrum.slice(0, usableFrames);
features.frameSize = usableFrames;
const predictions = await musicnn.predict(features, true);
```

改完 JS 從 80 → 79 patch，跟 Python 完全對齊。

---

## 修改前後對照

| 指標 | 修改前 | 修改後 |
|---|---|---|
| Python n_patches | 158 | **79** |
| JS n_patches | 80 | **79** |
| max \|diff\|（逐 tag 最大差） | 0.0057 | **0.0015** |
| Jaccard@10（Top-10 tags 重疊度） | 0.82 | **1.0000** |
| Spearman ρ（50 維相關係數） | 0.9985 | **0.9998** |
| n_patches 不一致 warning | 有 | **無** |
| 判定 | PASS | **PASS(strict)** |

兩端現在做的事情完全一樣：16kHz mono → mel-spectrogram → 79 個不重疊 patch → 對 patch 取平均。

---

## 還剩下的差異（可接受、預期內）

兩端分數還是有非常微小的差(`max |diff| ≈ 0.0015`)，這是**預期且無法消除**的，不是 bug：

- **mel-spectrogram 計算**：Python 用 essentia 的 C++ 實作，JS 用 essentia.js 編出來的 WASM。同樣演算法、不同 toolchain，浮點數會有微小差。
- **inference backend**：Python 走 libtensorflow(C++)，JS 刻意走純 JS 的 tfjs CPU backend（啟動時硬性 assert backend 是 `cpu`）。我們**故意不裝** `@tensorflow/tfjs-node`，避免 native backend 偷偷接管 —— 那會讓數值「太接近」而失去比對的意義（我們要模擬真實瀏覽器環境）。

判定 threshold 就是為了容許這種程度的差異：

- **嚴格 PASS**：`max |diff| < 0.05` **且** Top-10 Jaccard ≥ 0.8
- **寬鬆 PASS**：Top-10 Jaccard ≥ 0.6 且 Spearman ρ ≥ 0.9

目前是嚴格 PASS，而且遠在 threshold 之內。

---

## 結論

兩端在數值上已經高度一致（Top-10 完全相同、ρ = 0.9998），代表把 MSD-MusiCNN inference 搬到前端 / Node 是可行的，輸出可以信任。剩下的微小浮點差異是不同 toolchain 的本質差異，在判定 threshold 內，屬於可接受範圍。
