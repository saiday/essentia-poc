#!/usr/bin/env bash
set -e

# 1. Python venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Node deps
npm install

# 3. Download models from essentia model zoo
mkdir -p models
curl -L -o models/msd-musicnn-1.pb \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb
curl -L -o models/msd-musicnn-1.json \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json
curl -L -o /tmp/msd-musicnn-1-tfjs.zip \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1-tfjs.zip

# 注意:essentia 官方那份 tfjs zip 是用一個有 bug 的工具打的:
#   1. 檔名長度欄位 (fname_len) 被截斷,所以 unzip / ditto / bsdtar / python.zipfile
#      全部都會在「central directory 找不到 / bad magic」這關掛掉。
#   2. 每個 entry 的 "ux" (Unix uid/gid) extra field payload 11 bytes 沒被算進 extra_len。
# 解法:跳過所有標準 zip 工具,直接掃 PK\x03\x04 local file header,手動補回那 11 bytes 之後解壓。
# 用 content sniffing (JSON 開頭是 '{' 才當 model.json) 來分配檔名,因為 zip 裡的檔名本身是壞的。
mkdir -p models/msd-musicnn-1-tfjs
python3 - <<'PY'
import struct, zlib, os, json
ZIP = "/tmp/msd-musicnn-1-tfjs.zip"
OUT = "models/msd-musicnn-1-tfjs"
UX_PAD = 11
with open(ZIP, "rb") as f:
    data = f.read()
LFH = b"PK\x03\x04"
offsets, i = [], 0
while True:
    j = data.find(LFH, i)
    if j < 0: break
    offsets.append(j); i = j + 4
for off in offsets:
    sig, ver, flags, comp, mt, md, crc, csize, usize, fl, el = \
        struct.unpack("<IHHHHHIIIHH", data[off:off+30])
    if usize == 0: continue
    start = off + 30 + fl + el + UX_PAD
    blob = data[start:start+csize]
    raw = zlib.decompress(blob, -15) if comp == 8 else blob
    assert len(raw) == usize, f"size mismatch {len(raw)} vs {usize}"
    name = "model.json" if raw[:1] == b"{" else "group1-shard1of1.bin"
    path = os.path.join(OUT, name)
    with open(path, "wb") as g: g.write(raw)
    print(f"  extracted {path} ({len(raw)} bytes)")
PY

echo
echo "Setup done. Now: place an audio file at ./input.mp3 and run:"
echo "  bash run_test.sh ./input.mp3"
