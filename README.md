# ger-speech-hybrid-asr
Context-Aware German Speech Recognition
# ger-speech-hybrid-asr: Context-Aware German Speech Recognition using Wav2Vec2 & KenLM

A robust pipeline designed to evaluate and improve German Automatic Speech Recognition (ASR). This project bridges the gap between raw acoustic modeling and linguistic context by integrating a **Wav2Vec2 Acoustic Model** with a statistical **n-gram Language Model (KenLM)** using a customized CTC beam search decoder.

---

## 🎯 Project Goals

* **Mitigate Acoustic Typos:** Improve standard Connectionist Temporal Classification (CTC) decoding by utilizing linguistic context.
* **Normalize Raw Transcriptions:** Implement a text-normalization preprocess (using Regex and `num2words`) to map digits to written German form.
* **Compare Decoding Strategies:** Analyze the mathematical and practical trade-offs between ultra-fast Greedy Decoding and computationally intensive CTC-LM Beam Search Decoding.
* **Optimize for Real-World Audio:** Handle on-the-fly resampling of high-frequency FLEURS audio datasets down to the 16kHz standard expected by modern deep neural network architectures.

---

### Key Architecture  💡

* **The WER vs. CER Paradox:** By integrating the 3-gram KenLM, the **Character Error Rate (CER) dropped significantly from 18.30% to 17.44%**. This proves that the language model successfully corrects phonetic spelling errors and acoustic typos.
* **Out-of-Vocabulary (OOV) Bottleneck:** Because the language model was trained on a localized corpus (~3,000 sentences), it struggled with highly diverse words in the test set. In some cases, it overcorrected correct acoustic frames into phonetically close words from the training vocabulary, resulting in a slight increase in WER (from 0.6496 to 0.6749).
* **Decoder Beam-Width Sensitivity:** Elevating the beam width to 100 allows the system to evaluate alternate structural hypotheses, improving context correction at the cost of linearly increasing decoding latency.


---

## 🔄 Process Flow

```
[Raw FLEURS Dataset] 
         │
         ▼
 [Text Normalization]  ───► [Extract text corpus] ───► [Build 3-gram KenLM]
(Regex & num2words mapping)                                    │
         │                                                     │
         ▼                                                     ▼
[In-Memory Audio Bytes]                               [lm.arpa State File]
         │                                                     │
         ▼                                                     │
 [16kHz Resampling]                                            │
         │                                                     │
         ▼                                                     │
 [Wav2Vec2 Inference]  ────────────────────────► [pyctcdecode Beam Search]
 (Generate Frame Logits)                               (Combines Acoustic + LM)
                                                               │
                                                               ▼
                                                     [Final Transcriptions]

```

---

## 📊 Evaluation & Findings

Models were evaluated on the **German FLEURS test partition** using a Tesla T4 GPU.

### Performance Comparison

| Decoding Strategy | Word Error Rate (WER) | Character Error Rate (CER) | Latency / Speed |
| --- | --- | --- | --- |
| **Greedy Decoding (Baseline)** | **0.6496** | 0.1830 | **Instantaneous** (Frame-by-frame $\text{argmax}$) |
| **CTC + KenLM (Default Beam)** | 0.6749 | **0.1744** | Balanced (~4.89 samples/sec) |
| **CTC + KenLM (Beam Width = 100)** | *Tuned* | *Tuned* | Computationally intensive |


## 🚀 How to Run the Pipeline

### 1. Requirements Installation

```bash
pip install torch transformers pyctcdecode kenlm librosa soundfile evaluate tqdm

```

### 2. Prepare the Corpus & Train Language Model

Ensure your transcription dataset is normalized, then output your training sentences to `corpus.txt` with space-separated words:

```python
# Run KenLM compiler (requires cmake & boost libraries)
!lmplz -o 3 < corpus.txt > lm.arpa

```

### 3. Run Inference & Decode with LM

```python
from pyctcdecode import build_ctcdecoder
import soundfile as sf
import io

# Initialize decoder
decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path="lm.arpa",
    alpha=0.2,
    beta=0.8
)

# Decode output logits from Wav2Vec2
logits_numpy = logits[0].cpu().numpy()
prediction = decoder.decode(logits_numpy, beam_width=100)

```
