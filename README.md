# Multi-Party Conversation Modeling & Multimodal Visual Grounding

[![CI](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Hierarchical_BiLSTM_%2B_Vision--Language-informational.svg)](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems)
[![AMI Next-Speaker Accuracy](https://img.shields.io/badge/AMI_Next--Speaker_Acc-70.72%25-success.svg)](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems)
[![GuessWhat Grounding](https://img.shields.io/badge/GuessWhat_Validity-100%25_(Recoverable)-purple.svg)](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research and production-grade conversational AI toolkit dedicated to two core challenges in interactive systems:
1. **Multi-Party Conversational Modeling**: Tracking dynamic speaker floor control, addressee recognition, and next-speaker prediction in multi-participant group meetings using the AMI Meeting Corpus.
2. **Multimodal Visual Reference Grounding**: Grounding deictic and spatial linguistic expressions against visual candidates in shared workspaces using vision-language models on the GuessWhat?! benchmark.

---

## 1. Multi-Party Conversation Modeling & Next-Speaker Prediction

Group conversations follow distinct turn-taking dynamics that differ from two-party dialogue. In a meeting with participants $\{A, B, C, D\}$, who speaks next depends on conversational history, participant roles, and prior turn ownership, alongside the lexical content of the current utterance.

### 1.1 Temporal Turn Aggregation (`build_turns`)
Raw meeting transcriptions record words with millisecond start and end timestamps. Consecutive word fragments from the same participant are merged into single conversational turns whenever the inter-word gap satisfies:

$$\Delta t = t_{\mathrm{start}}^{(i)} - t_{\mathrm{end}}^{(i-1)} \le \tau_{\mathrm{gap}}, \qquad \tau_{\mathrm{gap}} = 0.5\text{ seconds}$$

### 1.2 Hierarchical Next-Speaker BiLSTM Network
To predict who will speak next ($y \in \{A, B, C, D\}$), the system employs a two-tier hierarchical recurrent architecture:

```
[Turn t-7] Utterance BiLSTM ──┐
[Turn t-6] Utterance BiLSTM ──┼──► Speaker Embeddings ──► Context BiLSTM ──► Next-Speaker Classifier (A, B, C, D)
...                           │
[Turn t]   Utterance BiLSTM ──┘
```

1. **Utterance-Level BiLSTM:** Each turn's token sequence is projected into $128$-dimensional word embeddings and processed via a bidirectional LSTM with packed sequences to produce a dense utterance summary $h_{\mathrm{utt}} \in \mathbb{R}^{256}$.
2. **Speaker Identity Embedding:** Categorical speaker IDs are projected into learned vectors $e_{\mathrm{spk}} \in \mathbb{R}^{16}$ and concatenated with $h_{\mathrm{utt}}$.
3. **Context-Level BiLSTM:** A sequence of up to $H = 8$ preceding turns is passed through a context bidirectional LSTM to capture multi-turn conversational momentum.
4. **Classification Head:** The context representation and current speaker vector are mapped through a two-layer MLP with ReLU and Dropout ($p = 0.3$) to yield logits over the candidate speakers.

### 1.3 Benchmark Results on AMI Meeting Corpus
Trained and evaluated on the canonical AMI Meeting Corpus across 20 full multi-party sessions:

| Model Variant | Dialogue Context | Speaker Embeddings | Test Accuracy | Macro F1 | Dev Peak F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Hierarchical BiLSTM** | **8-Turn History** | **Yes (16-D)** | **70.72%** | **0.7048** | **0.7199** |
| Ablation 1: Text History Only | 8-Turn History | No (Ablated) | 27.36% | 0.2726 | 0.2579 |
| Ablation 2: Turn + Speaker Only | 0-Turn History | Yes (16-D) | 33.42% | 0.2726 | 0.2772 |

**Key Findings:**
- Removing speaker embeddings causes a catastrophic $-43.22\%$ drop in Macro F1 (collapsing toward random guessing among 4 balanced participants). Knowing *who* said what is vastly more informative for turn-taking than lexical semantics alone.
- Truncating multi-turn history causes a $-43.22\%$ drop, proving that conversational momentum spans multiple dialogue turns.

### 1.4 AMI Confusion Matrix (Speaker Distribution)
```
            Predicted A   Predicted B   Predicted C   Predicted D
Actual A:      3747           506           466           357
Actual B:       584          3433           443           391
Actual C:       490           507          3191           368
Actual D:       426           387           356          2383
```

---

## 2. Multimodal Reference Grounding & Visual Dialogue

In collaborative visual environments (such as robotic manipulation, shared design tools, or assistive search), conversational agents must resolve ambiguous referring expressions like *"the one on the far left"* or *"the small green object behind the cup"*.

### 2.1 Multi-Agent GuessWhat?! Architecture
The system formulates visual reference resolution as a cooperative game between three distinct agents:
1. **Questioner Agent:** Inspects the scene and formulates discriminative yes/no questions to narrow down candidate objects.
2. **Oracle Agent:** Conditioned on the ground-truth target object and its 2D spatial bounding box, answers incoming questions strictly with `{Yes, No, N/A}`.
3. **Selector Agent:** Reads the accumulated dialogue history, scores candidate visual objects, and selects the final target.

### 2.2 2D Bounding-Box Spatial Normalisation
Bounding boxes $[x, y, w, h]$ are transformed into normalized coordinate ranges $[0, 999]$ compatible with modern vision-language models (e.g. Qwen-VL):

$$x_{\mathrm{norm}} = \left\lfloor \frac{x}{W_{\mathrm{img}}} \times 1000 \right\rfloor, \qquad y_{\mathrm{norm}} = \left\lfloor \frac{y}{H_{\mathrm{img}}} \times 1000 \right\rfloor$$

### 2.3 Evaluation of Questioner & Oracle Configurations
Evaluated across 20 diverse multi-object visual scenes:

| Agent Configuration | Questioner Policy | Oracle Supervision | Strict Validity | Recoverable Validity | Task Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Config B (Zero-Shot)** | Constrained Prefill (`"Is it "`, $T=0.7$) | 0-Shot Hyper-Constrained | 91.0% | **100.0%** | **50.0%** |
| **Config B (LoRA)** | Constrained Prefill (`"Is it "`, $T=0.7$) | LoRA Fine-Tuned ($r=32, \alpha=64$) | **100.0%** | **100.0%** | 45.0% |
| **Config C (Zero-Shot)** | Open-Ended Semantic ($T=0.5$) | 1-Shot In-Context Prompt | 76.0% | **100.0%** | 35.0% |
| **Config C (LoRA)** | Open-Ended Semantic ($T=0.5$) | LoRA Fine-Tuned ($r=32, \alpha=64$) | **100.0%** | **100.0%** | 30.0% |
| **Config A (Baseline)** | Constrained Prefill (`"Is it "`, $T=0.7$) | 1-Shot In-Context Baseline | 76.0% | **100.0%** | 20.0% |

**Key Findings:**
- **Constrained Prefilling vs Open-Ended Generation:** Constraining the Questioner with the prompt prefix `"Is it "` (Config B) boosts final task accuracy from $35\%$ to $50\%$, preventing the agent from asking overly complex multi-clause questions that confuse downstream scoring.
- **LoRA Oracle Compliance:** LoRA fine-tuning on the Oracle achieves $100.0\%$ strict formatting compliance (eliminating conversational filler like *"Sure, yes!"* or *"I think not"*), while zero-shot prompting requires regex response normalization to achieve $100\%$ recoverable validity.

---

## 3. Repository Structure

```
multimodal-multiparty-dialogue-systems/
├── docs/
│   └── assets/
├── src/
│   └── dialogue_systems/
│       ├── __init__.py
│       ├── multiparty.py             # Multi-party turn builder, dataset generator & BiLSTM network
│       └── multimodal_grounding.py   # 2D coordinate normalization, GuessWhat?! agents & fusion scorer
├── tests/
│   └── test_dialogue_systems.py      # Automated unit test suite
├── .gitignore
├── LICENSE                           # MIT License
├── pyproject.toml
└── requirements.txt
```

---

## 4. Quickstart & Usage

### Installation
```bash
git clone https://github.com/Ceviche98/multimodal-multiparty-dialogue-systems.git
cd multimodal-multiparty-dialogue-systems
pip install -r requirements.txt
pip install -e .
```

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License: see the [LICENSE](LICENSE) file for details.
