# Nubi Speech Recognition in Low-Resource Settings

| | |
|---|---|
| **Degree** | B.A. Computational Linguistics, University of Tübingen |
| **Author** | Mohamed Khattab |
| **Grade** | **1.3** ("sehr gut" / very good)<br><sub>German scale: 1.0 is the best grade, 4.0 the lowest pass</sub> |
| **Model** | XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) |
| **Language** | Nubi (ISO 639-3: `kcn`) |

## Overview

This repository accompanies my bachelor's thesis on automatic speech recognition (ASR) for **Nubi** (ISO 639-3: `kcn`), a severely low-resource Arabic-lexifier creole spoken in Uganda and Kenya. The thesis fine-tunes XLS-R 300M on spontaneous Nubi speech to find out how much transcribed data the model needs before it learns anything useful, and tests whether a two-phase curriculum can lower that amount. The curriculum follows the easy-to-hard principle of Bengio et al. (2009); in this low-resource setting it made results worse, not better, suggesting the principle depends heavily on how "easy" is defined.

---

## Research Questions

1. **Data threshold:** At what amount of transcribed speech does XLS-R 300M begin to learn Nubi meaningfully?
2. **Curriculum learning:** Can a prompt-to-transcription curriculum enable learning where standard fine-tuning struggles?

## Key Findings

- **A steep data threshold lies between 2.97 h and 5.94 h.** At 1 h the model fails completely (WER 1.0); at 2.97 h it learns only marginally; at 5.94 h error rates drop substantially.
- **The curriculum did not help, which runs counter to the easy-to-hard hypothesis of Bengio et al. (2009).** Bengio et al. argued that training on easier examples first acts as a continuation method that guides optimization toward better minima. I tested a curriculum built on that principle: short prompt labels first, then full transcriptions. At 2.97 h it performed worse than direct fine-tuning on both WER (0.760 vs 0.629) and CER (0.264 vs 0.230). Phase 1 drove training loss down while validation WER stayed at 1.0, so the "easy" phase taught the model nothing that transferred and used up epochs that direct fine-tuning spent learning. This adds a low-resource ASR case to the evidence (e.g. Wu et al., 2021) that curriculum benefits are not universal and depend on how "easy" is defined.
- **Vowel-height confusions dominate the errors** (a↔e, i→e, u↔o, i↔u) in every condition, including the largest data size and the curriculum model.
- **Conclusion:** data quantity is the main bottleneck for Nubi ASR. Curricula grounded in Nubi phonology are a more promising direction than ones based on utterance length or label complexity.

## Results

Test-set results, XLS-R 300M on Nubi:

| Condition | Training data | WER | CER |
|---|---|---|---|
| Baseline | 1 h | 1.000 | 1.000 |
| Baseline | 2.97 h | 0.629 | 0.230 |
| Curriculum | 2.97 h | 0.760 | 0.264 |
| Baseline | 5.94 h | 0.505 | 0.188 |

Learning onset (validation): around epoch 8 at 5.94 h; around epoch 17 at 2.97 h; never at 1 h. In the curriculum run, learning only began about 15 epochs into Phase 2.

---

## Data

- **Corpus:** Mozilla Common Voice Spontaneous Speech, Nubi, **version 2.0** (no longer distributed; the current release is 3.0 via the Mozilla Data Collective).
- **Speakers:** 26, with no speaker overlap between splits (verified by speaker-ID intersection).
- **After cleaning:** 1,885 clips, split into 1,135 train / 332 dev / 418 test.
- **Cleaning:** removed entries with missing audio or transcriptions, clips shorter than 2 s or longer than 60 s, and files that failed audio integrity checks.
- **Text normalization:** lowercasing, removal of non-alphabetic characters, whitespace normalization.
- **Vocabulary:** character-level, built from train and dev only (no test leakage), with a dedicated word-boundary token.
- **Audio:** resampled to 16 kHz and normalized with `Wav2Vec2FeatureExtractor`.

### Training subsets

Three **nested** subsets (1 h ⊂ 2.97 h ⊂ 5.94 h) are drawn by shuffling the training pool with seed 42 and adding clips until each target duration is reached. Nesting means differences in performance come from data quantity rather than clip selection. The nesting property is checked before training.

---

## Method

### Model

- `facebook/wav2vec2-xls-r-300m` with a randomly initialized linear CTC head over the Nubi character vocabulary.
- Convolutional feature encoder **frozen**; only the Transformer layers and output projection are trained.
- **SpecAugment disabled** to avoid masking away the limited training signal and to keep conditions directly comparable.

XLS-R was chosen over MMS because its weakness below about one hour of data is well documented, which makes it suitable for studying where learning begins and whether a curriculum can move that point.

### Baseline training

| Setting | Value |
|---|---|
| Batch size | 8 (× 2 gradient accumulation = 16 effective) |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 1e-4, cosine decay |
| Warmup | 5% of steps (minimum 50) |
| Gradient clipping | max norm 1.0 |
| CTC loss reduction | `mean` |
| Epochs | 40, no early stopping; best validation-WER checkpoint kept |
| Precision / memory | FP16, gradient checkpointing (`use_reentrant=False`) |
| Batching | `group_by_length=False` (random batches) |

Early stopping is disabled on purpose: very small datasets may need many passes before learning starts.

### Curriculum condition

Same data, clip IDs, vocabulary, optimizer and hyperparameters as the baseline; only the training procedure differs.

- **Phase 1 (10 epochs, LR 3e-4):** train on short prompt labels (about 11 words / 55 characters per clip).
- **Phase 2 (30 epochs):** continue on the full transcriptions (about 30 words / 149 characters per clip).

Prompts and transcriptions are genuinely different targets (mean word-level Jaccard overlap 0.116, no exact matches across 1,885 pairs). Clips stay randomly ordered within each phase; only label complexity changes between phases. As far as I know, this phase-level, label-complexity curriculum had not previously been tested in low-resource ASR.

### Evaluation

- **WER** (primary) and **CER**, both computed as (S + D + I) / N. A WER of 1.0 counts as complete failure.
- **Character-level error analysis** using Levenshtein edit operations to build substitution, insertion and deletion confusion matrices, reported only for checkpoints where the model has learned meaningful alignments.

---

## Reproducibility and Fairness Controls

- Fixed random seeds for Python, NumPy and PyTorch, with CUDA determinism enforced.
- MD5 hashes over clip IDs, transcriptions, prompts, speaker IDs, durations and audio paths; every field must match between baseline and curriculum before training proceeds.
- MD5 checks on vocabulary and processor files before each run.
- A fairness manifest (JSON) written before each run, recording dataset hashes, vocabulary hashes, total durations and all configuration parameters.

---

## Limitations and Future Work

- Only three data sizes were tested, so the threshold is located to a range (2.97 to 5.94 h), not a point.
- Only one curriculum design was evaluated, and only at 2.97 h.
- The original Common Voice 2.0 release is no longer available, so exact replication requires the archived copy.
- Promising next steps: data augmentation, and curricula organized around phonological difficulty (especially vowel contrasts) or speaker variability.

---

## Citation

```bibtex
@thesis{khattab2026nubi,
  author = {Khattab, Mohamed},
  title  = {Nubi Speech Recognition in Low-Resource Settings},
  school = {University of T{\"u}bingen},
  type   = {B.A. thesis},
  year   = {2026}
}
```

