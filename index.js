import { TokenizerWasm } from "hf-tokenizers-wasm";
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Adds the WASM backend to the global backend registry.
import '@tensorflow/tfjs-backend-wasm';
// Set the backend to WASM and wait for the module to be ready.
import {loadGraphModel} from '@tensorflow/tfjs-converter';

tf.setBackend('wasm').then(() => main());

class Tokenizer {
  constructor(json) {
    this.tokenizer = new TokenizerWasm(json);
  }

  static from_pretrained(name) {
    return fetch(`https://huggingface.co/${name}/resolve/main/tokenizer.json`)
      .then(response => response.text())
      .then(json => new Tokenizer(json));
  }

  encode(text) {
    return this.tokenizer.encode(text);
  }
}

const loadModel = async () => {
    try{
        const model = await loadGraphModel("https://raw.githubusercontent.com/finiteautomata/ner-leg-no-lfs/main/model.json");
        return model;
    }
    catch(error){
        console.log("There was an error loading the model!")
        console.log(error);
        throw error;
    }
}


async function main() {
    let tokenizer = await Tokenizer.from_pretrained("gpt2");
    let encoding = tokenizer.encode("I love AI and privacy", false);

    console.log("Loading model...");
    let model = await loadModel();
    console.log("done!");
    let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/data/flextronics.txt";

    let contractResponse = await fetch(url);
    let contract = await contractResponse.text();
    console.log(contract);
    console.log(encoding.input_ids);
    console.log(document);
    document.tokenizer = tokenizer;
}

main();