# ArNanoGPT 

A GPT-style language model for Arabic, built entirely from scratch — architecture, tokenizer, dataset, and training loop, no pretrained backbone.

Trained following the [nanoGPT](https://github.com/karpathy/nanoGPT) approach (Karpathy-style decoder-only Transformer), on a self-built Arabic Wikipedia corpus.

> This is a from-scratch pretrained model, not a fine-tune. Everything — the corpus, the byte-level BPE tokenizer, the transformer implementation, and the training loop — was built and trained by hand.

## Example output

| Before training (step 0, random weights) | After training (~35.6M params) |
|---|---|
| أحفمذ كوش Voرة عاني اثنصر انحلبرى ثية إلخالصان اس وغم ال بحتالمعة اتلأودوقت عان ر ابين... | في عام 1979 أصبحت أول امرأة تفوز بهذه الجائزة، وهي واحدة من بين أكثر عشرة جائزة تكريماً بعددها ثلاثين شخصاً... |

## Model details

| | |
|---|---|
| **Architecture** | Decoder-only Transformer (GPT-style), custom PyTorch implementation |
| **Tokenizer** | Byte-level BPE, vocab size 8,000, trained on the same Arabic corpus |
| **Parameters** | 35,638,592 (~35.6M) |
| **Context length (block_size)** | 256 tokens |
| **Layers** | 8 |
| **Attention heads** | 8 |
| **Embedding dimension** | 512 |
| **Dropout** | 0.2 |
| **Hardware** | Trained end-to-end on a single A100 GPU |

### Parameter breakdown

| Component | Parameters | % of total |
|---|---|---|
| Token embedding table | 4,096,000 | 11.5% |
| Position embedding table | 131,072 | 0.4% |
| Transformer blocks (×8) | 25,206,784 | 70.7% |
| Final LayerNorm | 1,024 | 0.0% |
| Feedforward (unused, legacy) | 2,099,712 | 5.9% |
| LM head | 4,104,000 | 11.5% |

## Repository structure

```
.
├── train.ipynb      # End-to-end training notebook: data prep, tokenizer training, model, training loop
├── inference.py      # Standalone script to load the trained model and generate text
└── README.md
```

The trained weights, tokenizer, and config are hosted on Hugging Face (see [Model weights](#model-weights) below) rather than committed to this repo.

## Training data

Trained on Arabic Wikipedia article text, collected and processed into a custom training corpus. The corpus has not been fully cleaned of wikitext template/markup artifacts (e.g. occasional leftover strings like `يجب تبديلها أو حذفها` or reference/citation placeholders), so the model can occasionally reproduce these as part of generated text.

## Getting started

### Requirements

```bash
pip install torch tokenizers huggingface_hub
```

### Model weights

The trained model (`pytorch_model.bin`), tokenizer (`tokenizer.json`), and config (`config.json`) are hosted on the Hugging Face Hub:

```
Hatim2221/arabic-nanogptv2
```

`inference.py` downloads these automatically via `huggingface_hub` — no manual download needed.

### Run inference

```bash
python inference.py
```

Or load it manually:

```python
from huggingface_hub import hf_hub_download
import torch, json
from tokenizers import Tokenizer

repo_id = "Hatim2221/arabic-nanogptv2"
config = json.load(open(hf_hub_download(repo_id, "config.json")))
tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, "tokenizer.json"))
weights_path = hf_hub_download(repo_id, "pytorch_model.bin")

# rebuild model architecture (see inference.py for the full class definitions), then:
# model.load_state_dict(torch.load(weights_path, map_location=device))
```

### Generation parameters

Recommended defaults for coherent output:

```python
output_ids = model.generate(
    context,
    max_new_tokens=300,
    temperature=0.7,
    top_k=40,
    repetition_penalty=1.3,
)
```

- `temperature` — lower = more conservative/coherent, higher = more random
- `top_k` — restricts sampling to the k most likely next tokens
- `repetition_penalty` — discourages the model from looping on the same word/phrase (recommended 1.2–1.5)

### Train it yourself

The full pipeline — corpus preparation, tokenizer training, model definition, and the training loop — is in [`train.ipynb`](./train.ipynb).

## Challenges along the way

A few real issues hit and fixed during development:

- **CUDA indexing errors** during training
- A **forward pass silently skipping half the network** — a subtle bug that trained without crashing but capped quality
- **Repetition loops** in generation, resolved with a repetition penalty combined with top-k/temperature sampling

## Intended use & limitations

This model is intended for **educational and experimental purposes** — exploring how small GPT models learn Arabic script, morphology, and Wikipedia-style prose structure.

- **Hallucination:** generates grammatically plausible but factually invented content (fake names, incorrect dates, invented events).
- **No factual grounding:** not suitable for question-answering, summarization of real events, or anything requiring accuracy.
- **Residual data artifacts:** may occasionally emit leftover Wikipedia template/markup text.
- **Small scale:** at ~35.6M parameters, coherence typically holds for a few sentences before drifting.

## Acknowledgements

Architecture and training approach inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## License

Add a license of your choice (e.g. MIT) if you intend for others to reuse this code.
