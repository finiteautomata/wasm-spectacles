import { TokenizerWasm } from "hf-tokenizers-wasm";

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

async function main() {
    let tokenizer = await Tokenizer.from_pretrained("gpt2");
    let encoding = tokenizer.encode("I love AI and privacy", false);

    let url = "https://raw.githubusercontent.com/finiteautomata/wasm-spectacles/master/data/flextronics.txt";

    let contractResponse = await fetch(url);
    let contract = await contractResponse.text();
    console.log(contract);
    console.log(encoding.input_ids);
    console.log(document);
    document.tokenizer = tokenizer;
}

main();