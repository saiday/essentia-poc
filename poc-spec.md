# Task: MSD-MusiCNN Cross-Implementation Parity Demo

## 目的

對同一份音檔 (`./input.mp3` 或使用者提供的路徑),分別用以下兩種方式跑 MSD-MusiCNN auto-tagging,
然後比對兩邊輸出的 50 維 MSD top-50 tag 機率是否一致。

- A. Python: `essentia.standard.TensorflowPredictMusiCNN`,模型檔 `msd-musicnn-1.pb`
- B. JavaScript: `essentia.js` 的 `EssentiaModel.TensorflowMusiCNN`,模型檔 `msd-musicnn-1-tfjs/model.json`

兩邊都只取模型「預設輸出」 (50 維 sigmoid,對應 MSD top-50 tags),不接任何下游分類頭。

## 不要做的事

1. 不要自己改 MusiCNN 的前處理參數 (frameSize / hopSize / melBands / sampleRate / patchSize)。
   essentia 兩端都已硬編好,擅自改參數會造成不對等。
2. 不要使用 `KEEP_PERCENTAGE` / `shortenAudio` 之類砍音檔的處理 (那是 essentia.js demo 為了即時性
   做的事,跟正確性無關)。整段音訊都要參與分析。
3. 不要用 essentia.js 的 `getAudioBufferFromURL` / `downsampleAudioBuffer` (Web Audio API 相依,
   在 Node 跑不了)。直接讀預處理好的 PCM。
4. 不要用 TFJS 的 WebGL backend (fp16 紋理會放大差異)。明確指定 CPU backend。
5. 不要把 tag 名稱從別的地方抄。**唯一可信來源是 `msd-musicnn-1.json` metadata 裡的 `classes` 欄位**。
   essentia.js 官方 demo 內嵌的字串列表有改過拼寫 (例如 `00s` → `2000s`),不要用那一份。

## 環境前提

- Python 3.10+ (essentia + TensorFlow)
- Node.js 18+ (essentia.js 0.1.3 + @tensorflow/tfjs,純 JS,不要 tfjs-node)
- ffmpeg (CLI)
- 一個任意音檔,放在 `./input.mp3` (或讓使用者帶路徑進來)

## 目錄結構 (執行完畢後應該長這樣)

```
.
├── input.mp3                          # 使用者提供的音檔
├── input_16k_mono.wav                 # 預處理結果,兩邊共用
├── models/
│   ├── msd-musicnn-1.pb               # Python 用
│   ├── msd-musicnn-1.json             # metadata (tag 名稱來源)
│   └── msd-musicnn-1-tfjs/            # JS 用
│       ├── model.json
│       └── group1-shard1of1.bin
├── python_tagger.py
├── js_tagger.js
├── compare.py
├── package.json
├── requirements.txt
├── results/
│   ├── python_output.json
│   ├── js_output.json
│   └── comparison_report.txt
└── README.md
```

## Step 1 — 環境與模型下載

寫 `setup.sh`,內容包含 (但不要實際執行 pip install / npm install,只寫好讓使用者執行;
Python 套件透過 venv 安裝):

```bash
# 1. 建 venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Node deps
npm install

# 3. 下載模型 (來自 essentia 官方 model zoo)
mkdir -p models
curl -L -o models/msd-musicnn-1.pb \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb
curl -L -o models/msd-musicnn-1.json \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json
curl -L -o /tmp/msd-musicnn-1-tfjs.zip \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1-tfjs.zip

# 注意:這個 zip 是 macOS 打的,含 AppleDouble metadata,unzip 會抱怨但能解。
# 解開後實際內容在 Users/.../msd-musicnn-1/ 路徑下,需要搬出來。
mkdir -p models/msd-musicnn-1-tfjs
cd /tmp && unzip -o msd-musicnn-1-tfjs.zip
# 把巢狀路徑下的 model.json + bin 搬到正確位置
find . -name "model.json" -path "*/msd-musicnn-1/*" -exec cp {} /workspace/models/msd-musicnn-1-tfjs/ \;
find . -name "*.bin" -path "*/msd-musicnn-1/*" -exec cp {} /workspace/models/msd-musicnn-1-tfjs/ \;
```

(把 `/workspace` 換成實際的專案根目錄)

`requirements.txt`:

```
essentia-tensorflow==2.1b6.dev1110
numpy
```

`package.json`:

```json
{
  "name": "msd-musicnn-parity-test",
  "version": "0.0.1",
  "type": "commonjs",
  "dependencies": {
    "essentia.js": "0.1.3",
    "@tensorflow/tfjs": "^4.10.0",
    "wav-decoder": "^1.3.0"
  }
}
```

## Step 2 — 音訊預處理 (兩端共用一份 PCM)

寫 `preprocess.sh`,用 ffmpeg 把任何輸入音檔轉成 16 kHz 單聲道 32-bit float WAV:

```bash
ffmpeg -y -i "$1" -ac 1 -ar 16000 -sample_fmt flt -c:a pcm_f32le input_16k_mono.wav
```

這份 `input_16k_mono.wav` 同時餵給 Python 和 JS,確保下游處理是從同一筆 PCM 樣本開始,
排除「兩邊解碼器/重採樣不同造成輸出差異」這個變因。

## Step 3 — Python 端 (`python_tagger.py`)

要求:

1. 用 `essentia.standard.MonoLoader` 載入 `input_16k_mono.wav`,sampleRate=16000,
   `resampleQuality=4` (預設,但明確寫出)。
2. 用 `TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb")`,
   `output` 參數**不指定** (拿 default output;明確不要設成 `model/dense/BiasAdd`,那是 embedding 不是 tag)。
3. 讀 `models/msd-musicnn-1.json` 取出 `metadata.classes` 作為 50 個 tag 名稱。
4. 輸出 shape 預期是 `(N_patches, 50)`。對 axis=0 取 mean,得到整首歌的 50 維平均機率向量。
   (這是 essentia 官方推薦的整曲層級彙整方式。)
5. 寫成 `results/python_output.json`:

```json
{
  "model": "msd-musicnn-1.pb",
  "audio": "input_16k_mono.wav",
  "n_patches": <int>,
  "tags": ["rock", "pop", "alternative", ...],   // 50 個,從 metadata 來
  "mean_activations": [0.123, 0.456, ...],       // 50 個,跟 tags 對齊
  "top10": [
    {"tag": "rock", "score": 0.812},
    ...
  ]
}
```

範例骨架:

```python
import json
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

audio = MonoLoader(filename="input_16k_mono.wav",
                   sampleRate=16000, resampleQuality=4)()

model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb")
activations = model(audio)  # shape (N, 50)

with open("models/msd-musicnn-1.json") as f:
    meta = json.load(f)
tags = meta["classes"]
assert len(tags) == 50

mean_act = activations.mean(axis=0)
top10_idx = np.argsort(mean_act)[::-1][:10]

out = {
    "model": "msd-musicnn-1.pb",
    "audio": "input_16k_mono.wav",
    "n_patches": int(activations.shape[0]),
    "tags": tags,
    "mean_activations": mean_act.tolist(),
    "top10": [{"tag": tags[i], "score": float(mean_act[i])} for i in top10_idx],
}
with open("results/python_output.json", "w") as f:
    json.dump(out, f, indent=2)
```

## Step 4 — JS 端 (`js_tagger.js`)

跑在 Node.js,不開瀏覽器。

要求:

1. 用 `wav-decoder` 讀 `input_16k_mono.wav`,拿到 `Float32Array` 樣本 (channel 0)。
   不要做任何重採樣或正規化 (檔案已經是 16 kHz mono)。
2. 載入 essentia.js WASM (Node 模式):

   ```js
   const { EssentiaWASM } = require("essentia.js/dist/essentia-wasm.umd.js");
   const wasm = await EssentiaWASM();   // or factory call as needed
   const Model = require("essentia.js/dist/essentia.js-model.umd.js").EssentiaModel;
   ```

   essentia.js 0.1.3 在 Node 端的入口名稱要去 `node_modules/essentia.js/dist/` 看實際 export,
   如果 UMD 在 Node 起不來,就改用 `essentia-wasm.es.js` + `essentia.js-model.es.js`,
   用 dynamic `import()`。

3. 設定 tfjs backend 為 CPU,並硬性 assert 切換成功:

   ```js
   const tf = require("@tensorflow/tfjs");
   // 故意不 require @tensorflow/tfjs-node。
   // 一旦 require 它,就會註冊並搶走 default backend (tensorflow native),
   // 而某些 tfjs 版本組合下 setBackend("cpu") 是 no-op,造成我們以為在跑 cpu 其實沒有。
   await tf.setBackend("cpu");
   await tf.ready();

   const actualBackend = tf.getBackend();
   console.log(`[backend check] requested=cpu, actual=${actualBackend}`);
   if (actualBackend !== "cpu") {
     throw new Error(
       `Backend assertion failed: wanted "cpu", got "${actualBackend}". ` +
       `Did something else register a backend before this script ran?`
     );
   }
   ```

   理由:本測試的目的是模擬「使用者在瀏覽器 / 純 JS 環境跑 essentia.js」會看到什麼結果。
   如果偷偷用了 TF native C++ backend,反而會跟 Python 端 (也是 TF native) 數值更接近,
   失去比對意義。inference 一首歌的 latency 用 pure JS cpu 完全可以接受,不需要 native 加速。

4. 算 mel-spectrogram patches:

   ```js
   const extractor = new Model.EssentiaTFInputExtractor(wasm, "musicnn", false);
   const features = extractor.computeFrameWise(audioFloat32, 256);   // hopSize=256
   ```

   `features` 物件含 `melSpectrum`, `frameSize`, `melBandsSize`, `patchSize`。

5. 載入 MusiCNN 模型並推論:

   ```js
   const modelURL = "file://" + path.resolve("models/msd-musicnn-1-tfjs/model.json");
   const musicnn = new Model.TensorflowMusiCNN(tf, modelURL);
   await musicnn.initialize();
   const predictions = await musicnn.predict(features, true);  // zeroPadding=true
   // predictions shape: [n_patches][50]
   ```

6. 用 `models/msd-musicnn-1.json` 的 `classes` (跟 Python 同一份來源) 拿 tag 名稱。

7. 對 patches 取 mean,輸出格式跟 Python 端**完全一致**:

```json
{
  "model": "msd-musicnn-1-tfjs",
  "audio": "input_16k_mono.wav",
  "n_patches": <int>,
  "tags": [...],
  "mean_activations": [...],
  "top10": [...]
}
```

存到 `results/js_output.json`。

### JS 端的踩雷檢查清單

- 啟動時 `tf.getBackend()` 必須等於 `"cpu"`,不等於就 throw (見 Step 4 第 3 點)。
- `tf.loadGraphModel(file://...)`:Node 端要用 `file://` URL,不能直接吃路徑。
- `EssentiaTFInputExtractor` constructor 第一個參數是 WASM module 物件,不是 factory。
- 確認 `predictions[0].length === 50`。如果不是 50,代表 model 載錯或拿到 embedding 層輸出。
- `zeroPadding=true` 跟 Python 預設行為一致 (essentia 那邊內部會 zero-pad 末端不足一個 patch 的部分)。

## Step 5 — 比對 (`compare.py`)

讀 `results/python_output.json` 與 `results/js_output.json`,輸出 `results/comparison_report.txt`,
包含以下檢查與數值:

1. **shape 與 tag 對齊檢查**
   - 兩邊都應該各有 50 個 tag,且 tag 名稱列表完全相同 (來自同一份 metadata)。
   - `n_patches` 兩邊應該相同。如果不同,印警告 (代表前處理沒對齊)。

2. **逐 tag 機率差異**
   - `diff = python.mean_activations - js.mean_activations`
   - 列出:`max |diff|`、`mean |diff|`、`std diff`
   - 列出 |diff| 最大的前 5 個 tag 與兩邊的數值。

3. **Top-K 集合比較**
   - K=5 與 K=10 兩種:
     - Jaccard similarity of top-K sets
     - Spearman rank correlation of 50 維機率向量

4. **判定**

   印出 PASS / FAIL,根據以下標準:
   - 嚴格 PASS: `max |diff| < 0.05` **且** Top-10 Jaccard ≥ 0.8
   - 寬鬆 PASS: Top-10 Jaccard ≥ 0.6 且 Spearman ρ ≥ 0.9
   - 否則 FAIL,印出可能原因 (前處理不一致、backend 不一致、模型版本不一致等)。

   報告裡明確寫出採用了哪個層級的 PASS。

5. **不要在 compare.py 裡自動「修正」差異** (例如 z-score、min-max scale 之類),
   sigmoid 機率應該就直接可比。

範例骨架:

```python
import json
import numpy as np
from scipy.stats import spearmanr

py = json.load(open("results/python_output.json"))
js = json.load(open("results/js_output.json"))

assert py["tags"] == js["tags"], "Tag lists differ!"
a = np.array(py["mean_activations"])
b = np.array(js["mean_activations"])
diff = a - b
absdiff = np.abs(diff)

top10_py = set(t["tag"] for t in py["top10"])
top10_js = set(t["tag"] for t in js["top10"])
jacc10 = len(top10_py & top10_js) / len(top10_py | top10_js)

rho, _ = spearmanr(a, b)

# ... 寫報告 ...
```

## Step 6 — 一鍵跑測試

寫 `run_test.sh`:

```bash
#!/usr/bin/env bash
set -e
INPUT="${1:-input.mp3}"
mkdir -p results
bash preprocess.sh "$INPUT"
source .venv/bin/activate
python python_tagger.py
node js_tagger.js
python compare.py
cat results/comparison_report.txt
```

## Step 7 — README.md

寫一份簡短的 README,內容:
- 一句話描述目的
- 怎麼跑 `setup.sh` → `run_test.sh ./input.mp3`
- 預期看到什麼 (top-10 tags + PASS/FAIL)
- 已知差異來源 (Python 走 essentia C++ 算 mel-spectrogram,JS 走 essentia.js WASM 算 mel-spectrogram;
  tfjs CPU backend 跟 TF 的 op 實作不同,即使權重相同也會有微小數值差異)

## 給 Claude Code 的最後注意事項

1. 寫完 setup.sh / preprocess.sh / run_test.sh 後**不要自己執行**,只把指令列出來給使用者複製。
   原因:模型下載要 ~10 MB+ 流量,且 essentia-tensorflow 安裝在某些環境會壞,讓使用者自己跑。
2. 但 `python_tagger.py`、`js_tagger.js`、`compare.py` **要寫到能 import 不會語法錯** (寫完跑
   `python -c "import ast; ast.parse(open('python_tagger.py').read())"` 和 `node --check js_tagger.js`)。
3. 如果你發現 essentia.js 0.1.3 的 Node 端 entry 點跟我寫的不一樣 (例如 `essentia.js-model.umd.js`
   在 Node require 時報錯),請去 `node_modules/essentia.js/package.json` 看 `main` / `exports`
   欄位,選實際能用的那個 entry。**回報你做了什麼修改**,不要默默改。
4. 如果 `unzip` 解 tfjs zip 報錯但能解出檔案,那是預期行為 (zip 內有 macOS AppleDouble metadata),
   忽略警告即可。
5. 完成後在最後一則訊息列出:
   - 所有產生的檔案路徑
   - 任何「按 spec 寫但你覺得可能有問題」的點,留給我看
