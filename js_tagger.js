// MSD-MusiCNN inference on the JS side (Node, pure-JS tfjs CPU backend).
// 故意不 require @tensorflow/tfjs-node — 我們要模擬瀏覽器 / 純 JS 環境。
const tf = require("@tensorflow/tfjs");
const wavDecoder = require("wav-decoder");
const fs = require("fs");
const path = require("path");

async function main() {
  // --- 1. backend assertion -------------------------------------------------
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

  // --- 2. load 16k mono float32 wav ----------------------------------------
  const wavBuf = fs.readFileSync("input_16k_mono.wav");
  const decoded = await wavDecoder.decode(wavBuf);
  if (decoded.sampleRate !== 16000) {
    throw new Error(
      `Expected 16000 Hz wav, got ${decoded.sampleRate}. preprocess.sh failed?`
    );
  }
  if (decoded.channelData.length < 1) {
    throw new Error("wav has no channels");
  }
  const audio = decoded.channelData[0]; // Float32Array
  console.log(`[wav] sampleRate=${decoded.sampleRate}, samples=${audio.length}`);

  // --- 3. load essentia.js WASM --------------------------------------------
  // essentia.js 0.1.3 在不同打包形式下 EssentiaWASM 有時是 factory function、
  // 有時已經是初始化好的 module instance。Runtime detect 一下,兩種都吃。
  const wasmEntry = require("essentia.js/dist/essentia-wasm.umd.js");
  const EssentiaWASMRaw =
    (wasmEntry && wasmEntry.EssentiaWASM) ? wasmEntry.EssentiaWASM : wasmEntry;
  let wasm;
  if (typeof EssentiaWASMRaw === "function") {
    wasm = await EssentiaWASMRaw();
    console.log("[essentia] WASM resolved via factory call");
  } else {
    wasm = EssentiaWASMRaw;
    console.log("[essentia] WASM resolved directly (already instantiated)");
  }

  // --- 4. load essentia.js model module ------------------------------------
  const modelEntry = require("essentia.js/dist/essentia.js-model.umd.js");
  const Model =
    (modelEntry && modelEntry.EssentiaModel) ? modelEntry.EssentiaModel : modelEntry;

  // --- 5. compute mel-spectrogram patches ----------------------------------
  // EssentiaTFInputExtractor(wasmModule, modelType, isLogMel?)
  // hopSize=256 跟 essentia C++ MusiCNN 預設一致。
  const extractor = new Model.EssentiaTFInputExtractor(wasm, "musicnn", false);
  const features = extractor.computeFrameWise(audio, 256);
  console.log(
    `[features] frameSize=${features.frameSize}, ` +
    `melBandsSize=${features.melBandsSize}, ` +
    `patchSize=${features.patchSize}`
  );

  // --- 6. load MusiCNN model ----------------------------------------------
  // 純 JS tfjs 在 Node 端沒有 file:// IOHandler (undici fetch 不支援 file://),
  // 而我們刻意不裝 @tensorflow/tfjs-node。所以自己讀檔組 modelArtifacts,
  // 用 tf.io.fromMemory() 載入,然後直接塞回 TensorflowMusiCNN instance。
  const modelDir = path.resolve("models/msd-musicnn-1-tfjs");
  const modelJSON = JSON.parse(fs.readFileSync(path.join(modelDir, "model.json"), "utf8"));
  const weightSpecs = modelJSON.weightsManifest.flatMap((m) => m.weights);
  const shardPaths = modelJSON.weightsManifest.flatMap((m) => m.paths);
  // concat shards in declared order
  const shardBuffers = shardPaths.map((p) => fs.readFileSync(path.join(modelDir, p)));
  const totalBytes = shardBuffers.reduce((a, b) => a + b.byteLength, 0);
  const weightData = new Uint8Array(totalBytes);
  let cursor = 0;
  for (const b of shardBuffers) {
    weightData.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), cursor);
    cursor += b.byteLength;
  }
  const modelArtifacts = {
    modelTopology: modelJSON.modelTopology,
    weightSpecs,
    weightData: weightData.buffer,
    format: modelJSON.format,
    generatedBy: modelJSON.generatedBy,
    convertedBy: modelJSON.convertedBy,
    signature: modelJSON.signature,
    userDefinedMetadata: modelJSON.userDefinedMetadata,
  };
  const preloadedModel = await tf.loadGraphModel(tf.io.fromMemory(modelArtifacts));
  console.log(`[model] loaded from memory (${shardPaths.length} shard(s), ${totalBytes} weight bytes)`);

  // 塞進 essentia.js 的 wrapper — 跳過它的 initialize() (那會去 fetch file://)
  const musicnn = new Model.TensorflowMusiCNN(tf, "noop://memory");
  musicnn.model = preloadedModel;
  musicnn.isReady = true;

  // --- 7. predict ----------------------------------------------------------
  // 對齊 Python essentia 的 lastPatchMode="discard":把 mel frames 截到 patchSize
  // 的整數倍,丟掉末端不足一個 patch 的部分。essentia.js 的 padding=false 不是
  // 「丟尾」而是「只吃單一 patch」(會 assert frameSize === patchSize),所以改用
  // 截斷 + padding=true(整數倍時不會真的補 0)來達成兩端 n_patches 一致。
  const patch = features.patchSize;
  const usableFrames = Math.floor(features.frameSize / patch) * patch;
  features.melSpectrum = features.melSpectrum.slice(0, usableFrames);
  features.frameSize = usableFrames;
  console.log(`[features] trimmed to ${usableFrames} frames (${usableFrames / patch} full patches)`);
  const predictions = await musicnn.predict(features, true);
  if (!Array.isArray(predictions) || predictions.length === 0) {
    throw new Error(`predict() returned empty/invalid: ${JSON.stringify(predictions).slice(0, 200)}`);
  }
  if (predictions[0].length !== 50) {
    throw new Error(
      `Expected 50 dims per patch, got ${predictions[0].length}. ` +
      `Wrong model output? (Should be sigmoid tag head, not embedding.)`
    );
  }
  const nPatches = predictions.length;
  console.log(`[predict] n_patches=${nPatches}, dims=${predictions[0].length}`);

  // --- 8. tag names ---  唯一可信來源是 metadata.classes ------------------
  const meta = JSON.parse(fs.readFileSync("models/msd-musicnn-1.json", "utf8"));
  const tags = meta.classes;
  if (!Array.isArray(tags) || tags.length !== 50) {
    throw new Error(`Expected 50 tags in metadata, got ${tags ? tags.length : "?"}`);
  }

  // --- 9. mean over patches ------------------------------------------------
  const meanAct = new Array(50).fill(0);
  for (let p = 0; p < nPatches; p++) {
    const row = predictions[p];
    for (let i = 0; i < 50; i++) {
      meanAct[i] += row[i];
    }
  }
  for (let i = 0; i < 50; i++) {
    meanAct[i] /= nPatches;
  }

  // --- 10. top 10 ----------------------------------------------------------
  const indexed = meanAct.map((s, i) => ({ tag: tags[i], score: s }));
  indexed.sort((a, b) => b.score - a.score);
  const top10 = indexed.slice(0, 10);

  // --- 11. write output ----------------------------------------------------
  if (!fs.existsSync("results")) {
    fs.mkdirSync("results", { recursive: true });
  }
  const out = {
    model: "msd-musicnn-1-tfjs",
    audio: "input_16k_mono.wav",
    n_patches: nPatches,
    tags: tags,
    mean_activations: meanAct,
    top10: top10,
  };
  fs.writeFileSync("results/js_output.json", JSON.stringify(out, null, 2));
  console.log(`Wrote results/js_output.json (n_patches=${nPatches})`);
  for (const e of top10) {
    console.log(`  ${e.tag.padStart(20)}: ${e.score.toFixed(4)}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
