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

/// TODO ESTO ESTA A MANO, ES LO QUE HAY
const maxLength = 512;
const PAD_IDX = 0;
const CLS_IDX = 101;
const SEP_IDX = 102;

const pad = (inputIds, attentionMask) => {
    inputIds = [CLS_IDX, ...inputIds, SEP_IDX];
    attentionMask = [0, ...attentionMask, 0];

    if (inputIds.length > maxLength) {
        inputIds = inputIds.slice(0, maxLength);
        attentionMask = attentionMask.slice(0, maxLength);
    }
    else {
        let dif = maxLength - inputIds.length;
        let padding = Array(dif).fill(0);
        inputIds = Array.from(inputIds).concat(padding);
        attentionMask = Array.from(attentionMask).concat(padding);
    }
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

const tokensToWord = (tokens) => {
    let word = "";
    for (let i = 0; i < tokens.length; i++) {
        if (i == 0) {
            word = tokens[i];
        } else {
            word += tokens[i].slice(2);
        }
    }
    return word;
}

const decode = (prediction, tokenizedInput) => {
    // Decode the prediction
    // First, get the prediction for each token

    let tokenPreds = prediction.argMax(1).arraySync();
    let wordIds = [-1, ...tokenizedInput.word_ids];
    let currentWordId = null;

    let currentTokens = [];
    let currentLabel = null;

    let wordAndLabels = [];

    for (let i = 1; (i < prediction.shape[0]) && (i < tokenizedInput.tokens.length); ++i) {
        // This is because unaligned tokenization
        let token = tokenizedInput.tokens[i-1];
        let pred = tokenPreds[i];
        let wordId = wordIds[i];

        if (wordId !== currentWordId) {
            // Starts new word
            if (currentWordId !== null)
                wordAndLabels.push([tokensToWord(currentTokens), currentLabel]);
            currentWordId = wordId;
            currentLabel = id2label[pred];
            currentTokens = [token];
        } else {
            currentTokens.push(token);
        }
    }

    return wordAndLabels;
}


async function main() {

    let tokenizer = await Tokenizer.from_pretrained("finiteautomata/ner-leg");

    console.log("Loading model...");
    let model = await loadModel();
    console.log("done!");
    let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/assets/flextronics.txt";
    let contract = await loadContract(url);

    let paragraphs = contract.split("\n").map(normalize).filter(line => line.length > 0);

    console.log("Tokenizing");
    let encodedParagraphs = paragraphs.map(paragraph => tokenizer.encode(paragraph));
    console.log("done!");

    document.tokenizer = tokenizer;
    document.contract = contract;
    document.paragraphs = paragraphs;
    document.encodedParagraphs = encodedParagraphs;

    console.log("Predicting");
    let predictions = encodedParagraphs.map(encoding => [encoding, predict(model, encoding)]);
    console.log("done!");

    for (let [encoding, prediction] of predictions) {
        console.log("==============================");
        console.log(decode(prediction, encoding));
    }
}

main();