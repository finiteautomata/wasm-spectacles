import { TokenizerWasm } from "hf-tokenizers-wasm";
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';


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

const maxLength = 512;
const PAD_IDX = 0;

const pad = (inputIds, attentionMask) => {
    let dif = maxLength - inputIds.length;
    let padding = Array(dif).fill(0);
    inputIds = Array.from(inputIds).concat(padding);
    attentionMask = Array.from(attentionMask).concat(padding);
    return {inputIds, attentionMask};
}


const predict = (model, encoding) => {

    let {inputIds, attentionMask} = pad(encoding.input_ids, encoding.attention_mask);
    inputIds = tf.tensor(
        inputIds, undefined, "int32"
    );
    attentionMask = tf.tensor(
        attentionMask, undefined, "int32"
    );

    let modelInput = {
        "input_ids": inputIds.reshape([1, -1]),
        "attention_mask": attentionMask.reshape([1, -1]),
    }

    return model.predict(modelInput).squeeze(0);
}

const id2label = [
    "O",
    "B-marker",
    "I-marker",
    "B-reference",
    "I-reference",
    "B-term",
    "I-term"
];


const normalize = (line) => {
    let ret = line.replaceAll("\t", " ");
    ret = ret.replaceAll("  ", " ");
    ret = ret.replaceAll("“", "\"");
    ret = ret.replaceAll("”", "\"");
    return ret;
}


const loadContract = async (url) => {
    let contractResponse = await fetch(url);
    let contract = await contractResponse.text();

    return contract;
}

async function main() {

    let tokenizer = await Tokenizer.from_pretrained("finiteautomata/ner-leg");
    console.log("Loading model...");
    let model = await loadModel();
    console.log("done!");
    let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/data/flextronics.txt";
    let contract = await loadContract(url);


    console.log(contract);
    console.log(document);

    let paragraphs = contract.split("\n").map(normalize).filter(line => line.length > 0);

;

    console.log("Tokenizing");
    let encodedParagraphs = paragraphs.map(paragraph => tokenizer.encode(paragraph));
    console.log("done!");

    document.tokenizer = tokenizer;
    document.contract = contract;
    document.paragraphs = paragraphs;
    document.encodedParagraphs = encodedParagraphs;
    document.tf = tf;

    console.log("Predicting");
    let predictions = encodedParagraphs.map(encoding => predict(model, encoding));
    console.log("done!");
}

main();