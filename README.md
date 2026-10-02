# English-to-French Seq2Seq Translation with PyTorch

Educational Deep Learning / NLP project implementing a word-level English-to-French machine translation model with a PyTorch LSTM encoder-decoder architecture.

## Overview

This academic notebook demonstrates the main steps of neural machine translation: preparing paired sentences, building vocabularies, training an encoder-decoder, and generating French text. The original educational implementation is preserved.

Training and inference logic are implemented. Translation quality was not independently validated during repository cleanup, and no final translation benchmark is published. The dataset is not included.

## Features

- English and French vocabulary construction using lowercase and whitespace tokenization.
- Special tokens: `<pad>`, `<unk>`, `<sos>`, and `<eos>`.
- Sequence truncation and padding.
- An 80/20 train-validation split and `TensorDataset` / `DataLoader` preparation.
- LSTM encoder, LSTM decoder, and composed `Seq2Seq` model.
- Teacher-forced training with `CrossEntropyLoss` ignoring padding.
- Adam optimization and gradient clipping.
- Validation loss and perplexity calculation.
- Weight checkpoint saving when validation loss improves.
- Greedy decoding and reconstruction of French text.

## Architecture

```text
English sentence
        ↓
Lowercase + tokenization
        ↓
English vocabulary → token IDs
        ↓
Padding / truncation
        ↓
Embedding
        ↓
Encoder LSTM
        ↓
Hidden + Cell states
        ↓
Decoder Embedding
        ↓
Decoder LSTM
        ↓
Linear layer
        ↓
French vocabulary scores
        ↓
Greedy token selection
        ↓
French sentence
```

The model is a word-level Seq2Seq architecture. It does not use attention, Transformer layers, or a pretrained LLM. During training, the decoder receives ground-truth target tokens; during inference, it receives its previous generated token, starting with `<sos>`.

## Dataset

Supply a CSV named `data_seq2seq.csv` separately, with these columns:

| Column | Meaning |
|---|---|
| `en` | English source sentence |
| `fr` | Corresponding French target sentence |

The dataset is not distributed because its provenance, license, and redistribution rights are not confirmed. No dataset rows are reproduced here. Use data you are authorized to access.

The original notebook expects the file at:

```text
/content/drive/MyDrive/data_seq2seq.csv
```

## Data Preprocessing

The notebook lowercases text and splits it on whitespace. Each vocabulary is built from word frequencies, with a cap of 100 entries including four reserved tokens:

| Token | ID | Purpose |
|---|---:|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown word |
| `<sos>` | 2 | Start of sequence |
| `<eos>` | 3 | End of sequence |

Sentences are truncated to reserve space for boundary tokens, then padded on the right. Source sequences have length 7 and target sequences have length 6, allowing at most five source words and four target words.

Vocabularies are constructed before the 80/20 split. The split uses `random_state=1`. Encoded pairs become integer tensors and are batched using PyTorch data loaders.

## Model Configuration

| Setting | Value |
|---|---|
| Embedding dimension | 32 |
| Hidden dimension | 64 |
| Encoder LSTM layers | 2 |
| Decoder LSTM layers | 2 |
| Vocabulary cap | 100 tokens per language |
| Source maximum sequence length | 7, including boundary tokens |
| Target maximum sequence length | 6, including boundary tokens |
| Validation split | 20% |
| Batch size | 3 |
| Learning rate | 0.001 |
| Epochs | 10 |
| Gradient clipping maximum norm | 1.0 |
| Optimizer | Adam |
| Loss | `CrossEntropyLoss(ignore_index=pad_id)` |
| Device | CUDA when available, otherwise CPU |

## Training

The decoder input is `tgt[:, :-1]`, and the expected output is `tgt[:, 1:]`. The training loop computes cross-entropy loss, performs backpropagation, clips gradients, and updates weights using Adam.

Validation runs without gradient tracking. Loss is aggregated by the number of non-padding target tokens, and validation perplexity is calculated as `exp(val_loss)`.

When validation loss improves, the notebook saves `model.state_dict()` to `best_seq2seq.pt` in the working directory. The checkpoint contains weights only; vocabularies and configuration are not saved with it. The best checkpoint is not automatically reloaded for inference.

No training was run during repository cleanup, and successful end-to-end training is not claimed.

## Inference

`translate_sentence` initializes the decoder from the encoder's hidden and cell states. Starting with `<sos>`, it selects the highest-scoring token at each step and feeds that token into the next step. Generation stops at `<eos>` or the generation limit, which defaults to 15 steps.

`tokens_to_sentence` maps IDs back to words, removes start/end/padding tokens, and joins the remaining words. The example uses the current in-memory model and single-sentence greedy decoding.

## Technology Stack

- Python and PyTorch.
- pandas for CSV loading.
- scikit-learn for the train-validation split.
- tqdm for training progress.
- torchinfo for model inspection.
- Jupyter notebook format and Google Colab, including its Google Drive integration.

## Project Structure

```text
pytorch-en-fr-seq2seq/
├── english_french_seq2seq_pytorch.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset and generated model checkpoints are excluded from Git.

## Installation

The notebook's original execution environment is Google Colab. `requirements.txt` lists its imported third-party dependencies without inventing historical versions:

```bash
pip install -r requirements.txt
```

This is an instruction for a future run; dependencies were not installed during cleanup. Google Colab supplies `google.colab`, so it is not listed as a separate dependency. Installing the listed packages locally does not provide the notebook's Colab-specific Drive integration.

## Running in Google Colab

1. Open `english_french_seq2seq_pytorch.ipynb` in Google Colab.
2. Supply your separately obtained CSV in Google Drive at `MyDrive/data_seq2seq.csv`.
3. Ensure the dependencies listed in `requirements.txt` are available. The notebook contains a commented installation command for `torchinfo`.
4. Execute cells in order and authorize the Google Drive mount when prompted.
5. Continue through preprocessing, model construction, training, and the inference example when you intend to run the exercise.

The notebook mounts Google Drive at `/content/drive` and reads the configured CSV path. Executing training writes `best_seq2seq.pt`; it may replace an existing file with that name. Repository preparation did not mount Drive, load external data, execute inference, or generate a checkpoint.

## Limitations

- The dataset is not included; its provenance and license are not confirmed.
- Dependency versions are not pinned.
- The workflow assumes Google Colab and Google Drive.
- Vocabulary is small and word-level.
- Sequence lengths are short and fixed; long sentences are truncated.
- The encoder processes padded positions.
- There is no attention mechanism or Transformer.
- Inference uses greedy decoding.
- No BLEU, chrF, or comparable translation-quality evaluation is implemented.
- The best checkpoint contains weights only, without vocabularies or configuration.
- The best checkpoint is not automatically reloaded for inference.
- There are no automated software tests.
- There is no API or deployment implementation.

## Reproducibility

The repository is partially reproducible: the implementation is supplied, but the dataset must be obtained separately and dependencies are unpinned. Only the data-split seed is fixed; model initialization and shuffled training do not have complete random-seed control. Runtime and hardware differences may affect results.

Saved outputs and runtime-specific notebook metadata were removed for publication without regenerating results. The embedded architecture image was removed because publication rights were uncertain. All code cells and historical model settings were preserved.

## Academic Context

This is an educational academic NLP project implementing neural machine translation with Seq2Seq and LSTM. Generic academic context in the notebook is preserved. A course heading mentioning generative AI is not used to claim additional model capabilities.

## Author

**Bassem Wali**

Computer Engineering Student

GitHub: https://github.com/bassem2002
