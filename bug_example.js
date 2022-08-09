// Import @tensorflow/tfjs or @tensorflow/tfjs-core
const tf = require('@tensorflow/tfjs');
const {loadGraphModel} = require('@tensorflow/tfjs-converter');
// Add the WASM backend to the global backend registry.
require('@tensorflow/tfjs-backend-wasm');
//const {setWasmPaths} = require('@tensorflow/tfjs-backend-wasm');

tf.setBackend('cpu').then(() => main());

async function main() {
    console.log("Loading model...");
    const model = await loadGraphModel("https://raw.githubusercontent.com/finiteautomata/ner-leg-no-lfs/main/model.json");

    console.log("Done loading");
    inputIds = tf.tensor(
        Array(512).fill(1), undefined, "int32"
    );
    attentionMask = tf.tensor(
        Array(512).fill(1), undefined, "int32"
    );

    let modelInput = {
        "input_ids": inputIds.reshape([1, -1]),
        "attention_mask": attentionMask.reshape([1, -1]),
    }

    return model.predict(modelInput).squeeze(0);
}
