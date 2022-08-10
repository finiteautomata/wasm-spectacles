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

    if (currentWordId !== null)
        wordAndLabels.push([tokensToWord(currentTokens), currentLabel]);

    return wordAndLabels;
}

const bioToSegments = (wordAndLabels) => {

    let segments = [];
    let currentWords = [];
    let currentType = "";

    for (let [word, label] of wordAndLabels) {
        if (label === 'O'){
            if (currentType == "text")
                currentWords.push(word)
            else {
                if (currentWords.length > 0)
                    segments.push({
                        "text": currentWords.join(" "),
                        "type": currentType
                    })
                currentWords = [word]
                currentType = "text"
            }
        }
        else if(label[0] === 'B') {
            if (currentWords.length > 0)
                segments.push({
                    "text": currentWords.join(" "),
                    "type": currentType
                })
            currentWords = [word]
            currentType = label.slice(2)
        }
        else  {
            if (currentType === label.slice(2))
                currentWords.push(word);
            else {
                if (currentWords.length > 0)
                    segments.push({
                        "text": currentWords.join(" "),
                        "type": currentType
                    })
                currentWords = [word]
                currentType = label.slice(2)
            }
        }
    }
    if (currentWords.length > 0)
        segments.push({
            "text": currentWords.join(" "),
            "type": currentType
        })

    return segments
};

async function main() {

    let tokenizer = await Tokenizer.from_pretrained("finiteautomata/ner-leg");

    console.log("Loading model...");
    let model = await loadModel();
    console.log("done!");
    //let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/assets/contracts/flextronics.txt";
    let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/assets/contracts/0000009984-20-000109%3Aexh102form8-kamendment.txt";
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

    for (let encoding of encodedParagraphs) {
        console.log("==============================");
        let prediction = predict(model, encoding);
        let decoded = decode(prediction, encoding);
        let segments = bioToSegments(decoded);
        console.log(segments);
    }
    console.log("done!");
}

main();