
# Arabic NanoGPT v2

A small GPT-style character/subword language model trained from scratch on Arabic Wikipedia text, built following the [nanoGPT](https://github.com/karpathy/nanoGPT) architecture (Karpathy-style decoder-only Transformer).

This is a **from-scratch pretrained model**, not a fine-tune of an existing LLM — it was trained entirely on the uploader's own data and compute for learning/experimentation purposes.

## Model Details

- **Architecture:** Decoder-only Transformer (GPT-style), custom PyTorch implementation
- **Tokenizer:** Byte-level BPE, vocab size 8,000, trained on the same Arabic corpus
- **Parameters:** 35,638,592 (~35.6M)
- **Model size:** 135.95 MB (float32) / 67.98 MB (float16)
- **Context length (block_size):** 256 tokens
- **Layers:** 8
- **Attention heads:** 8
- **Embedding dimension:** 512
- **Dropout:** 0.2

### Parameter breakdown

| Component                  | Parameters  | % of total |
|-----------------------------|------------:|-----------:|
| Token embedding table       |  4,096,000  |    11.5%   |
| Position embedding table    |    131,072  |     0.4%   |
| Transformer blocks (x8)     | 25,206,784  |    70.7%   |
| Final LayerNorm             |      1,024  |     0.0%   |
| Feedforward (unused, legacy)|  2,099,712  |     5.9%   |
| LM head                     |  4,104,000  |    11.5%   |

## Training Data

Trained on Arabic Wikipedia article text. Note: the corpus has not been fully cleaned of wikitext template/markup artifacts (e.g. occasional leftover strings like "يجب تبديلها أو حذفها" or reference/citation placeholders), so the model can occasionally reproduce these as part of generated text.

## Intended Use

This model is intended for **educational and experimental purposes** — exploring how small GPT models learn Arabic script, morphology, and Wikipedia-style prose structure. It is **not suitable for factual use**: like any small language model, it generates fluent-looking text without any grounding in truth, and it will confidently produce plausible-sounding but fabricated names, dates, and events.

## Limitations

- **Hallucination:** Generates grammatically plausible but factually invented content (fake names, incorrect dates, invented events).
- **No factual grounding:** Do not use for question-answering, summarization of real events, or any application requiring accuracy.
- **Residual data artifacts:** May occasionally emit leftover Wikipedia template/markup text.
- **Small scale:** At ~35.6M parameters, coherence typically holds for a few sentences before drifting.

## How to Use

See `inference.py` for a complete, ready-to-run script. Quick example:

```python
from huggingface_hub import hf_hub_download
import torch, json
from tokenizers import Tokenizer

repo_id = "Hatim2221/arabic-nanogptv2"
config = json.load(open(hf_hub_download(repo_id, "config.json")))
tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, "tokenizer.json"))
weights_path = hf_hub_download(repo_id, "pytorch_model.bin")

# rebuild model architecture (see inference.py for full class definitions), then:
# model.load_state_dict(torch.load(weights_path, map_location=device))
```

Full working example with model definitions, generation with temperature/top-k/repetition penalty: see `inference.py` in this repo.

## Generation Parameters

Recommended defaults for coherent output:

```python
model.generate(
    context,
    max_new_tokens=300,
    temperature=0.7,
    top_k=40,
    repetition_penalty=1.3,
)
```

- `temperature`: lower = more conservative/coherent, higher = more random
- `top_k`: restricts sampling to the k most likely next tokens
- `repetition_penalty`: discourages the model from looping on the same word/phrase (recommended 1.2–1.5)

## Files in this repository

- `pytorch_model.bin` — model weights (state_dict)
- `config.json` — architecture hyperparameters
- `tokenizer.json` — trained BPE tokenizer
- `inference.py` — standalone script to load the model and generate text
