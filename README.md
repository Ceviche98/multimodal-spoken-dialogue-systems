# Spoken, Multi-Party & Multimodal Dialogue Systems

[![CI](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/Ceviche98/multimodal-spoken-dialogue-systems/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-grade research engineering framework and empirical investigation of Spoken Dialogue ASR Error Propagation, Decision-Theoretic Clarification Dialogue Management, Multi-Party Turn-Taking on the AMI Corpus, Multimodal Referring Expression Comprehension with Vision-Language Models (`Qwen3.5-0.8B` + LoRA), Hierarchical Dialogue Act Tagging, and Residual Attention-Equipped Seq2Seq Generation.**

---

## Abstract & Executive Summary

Modern conversational agents operate far beyond clean, dyadic text interfaces: they must transcribe accented speech in real time, decide when to clarify noisy or out-of-vocabulary constraints before corrupting dialogue state, predict turn transitions across multi-party meetings, and ground ambiguous referring expressions in cluttered visual scenes.

This repository presents a unified, modular Python toolkit (`dialogue_systems`) and comprehensive empirical study across **six core pillars of spoken, multi-party, and multimodal dialogue systems**:

| Pillar | Task & Benchmark | Core Architecture / Methodology | Headline Quantitative Finding |
| :--- | :--- | :--- | :--- |
| **1. Spoken ASR & RAG Resilience** | FLEURS (`en_us`) & 20-Utterance Accented Multi-Domain Benchmark (10 domains) | `Whisper-Base` vs. `Distil-Large-v3` vs. `Whisper Large-v3-Turbo` + Levenshtein WER/CER & Hybrid Semantic Similarity | **Turbo achieves 1.77% WER** (85% zero-error); **Distil-Large-v3 is Pareto-optimal** (**7.51% WER**, **145.3 ms latency**, **2.67% RTF**) |
| **2. Clarification Dialogue Management** | 12-Case Noisy Slot-Filling Benchmark (`M1–M3`, `A1–A2`, `O1–O3`, `N1–N4`) | Hierarchical Slot Normalizer (Exact $\to$ Substring $\to$ `difflib` N-Gram Fuzzy) + Confidence-Thresholded Grounding Policy ($\tau_{\text{low}}=0.84, \tau_{\text{med}}=0.94$) | **91.67% Policy Accuracy (11/12)**; formal proof of the **Token-Level Threshold Contradiction** (`A2` vs. `N3`) |
| **3. Multi-Party Turn-Taking** | AMI Meeting Corpus (147 4-speaker meetings, 986,960 words $\to$ 196,159 sliding-window instances) | Hierarchical `UtteranceBiLSTMEncoder` (packed sequences) + 16-D Speaker Identity Embeddings + Context BiLSTM (`HISTORY = 8`) | **70.72% Test Acc / 70.48% Macro-F1**; drops to **27.36% Acc** without speaker IDs and **33.42% Acc** without history |
| **4. Multimodal VLM Grounding** | GuessWhat?! Cooperative Visual Game on MS-COCO (3 Agents: Questioner, Oracle, Selector) | `Qwen/Qwen3.5-0.8B` with $[0, 999]$ normalized 2D bounding boxes + LoRA PEFT ($r=32, \alpha=64$ on `q,k,v,o_proj`) | **50.0% Peak Game Accuracy** (`Config B`); LoRA raises Oracle strict format validity from **76% $\to$ 100%**, revealing **Asymmetric Multi-Agent Breakdown** |
| **5. Dialogue Act Classification** | Switchboard Dialogue Act Corpus (`SwDA`, 223,606 utterances, 43 DAMSL tags) | Stacked BiLSTM with Class-Balanced Cross-Entropy vs. Dual-Path Parallel `CNN_BiLSTM` ($k \in \{3,4,5\}$) | `CNN_BiLSTM` reaches **70.63% Test Acc** while recovering **37.74% recall** on ultra-rare non-understanding repair (`br`, 0.13% frequency) |
| **6. Neural Seq2Seq Generation** | Cornell Movie-Dialogs (89,924 pairs) & HybridDialogue Wikipedia QA (4,359 pairs) | Frozen 50-D GloVe + BiGRU Encoder + Additive Bahdanau Attention + Residual `Add & Norm` Two-Layer GRU Decoder (`DecoderAdvanced`) | Residual `Add & Norm` achieves **0.0830 Loss** at Epoch 50 — a **15$\times$ reduction** over 1-layer baseline (**1.2123**) and **25$\times$** over naive 2-layer (**2.0400**) |

---

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Acoustic["1. Spoken Dialogue Front-End (spoken_asr.py)"]
        A["Accented User Speech (16 kHz PCM)"] --> B["Whisper ASR Front-End<br/>(Base / Distil-Large-v3 / Large-v3-Turbo)"]
        B --> C["Transcript + Acoustic Confidence<br/>WER / CER / RTF / Semantic Similarity"]
    end

    subgraph NLU_DM["2. NLU & Uncertainty-Aware Dialogue Management (dialogue_act.py, clarification_dm.py)"]
        C --> D["Dual-Path CNN+BiLSTM DA Tagger<br/>43 Switchboard DAMSL Acts (sd, b, qy, qw, br, bf)"]
        C --> E["Hierarchical Slot Normalizer<br/>Exact (1.0) → Substring (0.95) → Fuzzy (≥0.72) → OOV (0.0)"]
        E --> F{"Confidence Grounding Policy<br/>τ_low = 0.84, τ_med = 0.94"}
        F -->|"c ≥ 0.94 (High)"| G["Optimistic State Commit & DB Query"]
        F -->|"0.84 ≤ c < 0.94 (Medium)"| H["Implicit Clarification<br/>(Embed guess in next slot wh-query)"]
        F -->|"c < 0.84 or Mentioned OOV"| I["Explicit Clarification<br/>(Binary confirm or list ontology options)"]
    end

    subgraph MultiParty["3. Multi-Party Turn-Taking (multiparty.py)"]
        C --> J["Word-to-Turn Aggregator (MAX_GAP = 0.5s)<br/>8-Turn Sliding Window + Speaker IDs {A, B, C, D}"]
        J --> K["Hierarchical NextSpeakerBiLSTMModel<br/>Utterance BiLSTM (256-D) + Speaker Emb (16-D) → Context BiLSTM"]
        K --> L["Floor-Hold vs. Turn-Switch Prediction<br/>(70.72% Acc / 70.48% Macro-F1)"]
    end

    subgraph Multimodal["4. Multimodal VLM Referent Grounding (multimodal_grounding.py)"]
        M["MS-COCO Visual Scene + Candidate BBoxes [0, 999]"] --> N["Questioner Agent (Qwen3.5-0.8B)<br/>Config B ('Is it ' Prefill) vs. Config C (Open Semantic)"]
        N --> O["Oracle Agent (Zero-Shot vs. LoRA r=32)<br/>Strict Binary Grounding {Yes, No, N/A}"]
        O --> P["Selector Agent (Greedy '[' Prefill)<br/>Posterior Referent Resolution"]
    end

    subgraph NLG["5. Neural Response Generation (seq2seq_generator.py)"]
        G & H & I --> Q["BiGRU Encoder + Bahdanau Attention<br/>+ Residual Add & LayerNorm 2-Layer GRU Decoder"]
    end
```

---

## 1. Spoken Dialogue Systems & ASR Error Propagation (`spoken_asr.py`)

### 1.1 Mathematical Formulation of Acoustic & Semantic Metrics

Given a ground-truth reference word sequence $R = (r_1, \dots, r_{N_w})$ and character sequence $R_c = (c_1, \dots, c_{N_c})$, and an ASR hypothesis $H = (h_1, \dots, h_{M_w})$, we compute **Word Error Rate (WER)** and **Character Error Rate (CER)** via dynamic programming Levenshtein alignment:

$$\text{WER}(R, H) = \frac{S_w + D_w + I_w}{N_w}, \qquad \text{CER}(R_c, H_c) = \frac{S_c + D_c + I_c}{N_c}$$

where $S, D, I$ denote minimum-cost substitutions, deletions, and insertions. For real-time spoken dialogue deployment, computational efficiency is governed by the **Real-Time Factor (RTF)**:

$$\text{RTF} = \frac{T_{\text{inference}}}{T_{\text{audio}}}, \qquad \text{where } \text{RTF} \ll 1.0 \text{ is required for sub-turn responsiveness.}$$

Because surface WER treats all lexical substitutions equally—penalizing benign morphological variants (`"recognizing"` vs. `"recognising"`) identically to catastrophic entity corruptions (`"dense vectors"` $\to$ `"10 spectars"`)—we pair WER/CER with **Semantic Similarity** $\text{Sim}_{\text{sem}}(R, H) \in [0, 1]$ to measure downstream dense retrieval (RAG) resilience.

### 1.2 Baseline Scaling Laws on Google FLEURS (`en_us`)

Prior to domain-specific accented evaluation, we benchmarked four standard OpenAI Whisper checkpoints on a held-out sample of the **Google FLEURS (`en_us`)** test split:

| Whisper Checkpoint | Parameter Scale | FLEURS `en_us` WER (%) | Relative WER Reduction vs. `tiny` |
| :--- | :---: | :---: | :---: |
| `openai/whisper-tiny` | 39M | **15.02%** | — |
| `openai/whisper-base` | 74M | **10.66%** | $-29.0\%$ |
| `openai/whisper-small` | 244M | **6.96%** | $-53.7\%$ |
| `openai/whisper-medium` | 769M | **5.13%** | $-65.8\%$ |

### 1.3 Custom 20-Utterance Multi-Domain Accented Speech Benchmark

Standard benchmarks like FLEURS under-represent technical jargon, financial acronyms, and code-switched regional entities spoken by non-native speakers. We constructed a **20-utterance controlled benchmark** recorded by a **South American Spanish-native speaker** across **10 specialized domains** (`Academic`, `Business`, `Casual`, `Customer_Service`, `Data_Eng`, `Finance`, `Food_Peru`, `History_War`, `Support`, `Tech_NLP`) and evaluated three architectures representing distinct deployment paradigms:

1. **`Whisper-Base` (`openai/whisper-base`, 74M)**: Lightweight edge baseline.
2. **`Distil-Whisper` (`distil-whisper/distil-large-v3`, 756M)**: Knowledge-distilled encoder-decoder with a shallow 2-layer decoder for low-latency streaming.
3. **`Whisper Large-v3-Turbo` (`openai/whisper-large-v3-turbo`, 809M)**: Full 32-layer encoder paired with a pruned 4-layer decoder trained on 5M+ hours of multilingual/accented audio.

| Architecture | Mean WER (%) | Median WER (%) | Max WER (%) | Zero-Error Utterances (%) | Mean CER (%) | Mean Semantic Similarity (%) | Mean Latency (ms) | Mean RTF (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Whisper-Base`** | 15.24% | 17.42% | 40.00% | 30.0% (6/20) | 5.80% | 89.62% | 266.05 ms | 4.93% |
| **`Distil-Whisper (distil-large-v3)`** | **7.51%** | **8.39%** | **22.22%** | **45.0% (9/20)** | **2.37%** | **94.83%** | **145.34 ms** | **2.67%** |
| **`Whisper Large-v3-Turbo`** | **1.77%** | **0.00%** | **16.67%** | **85.0% (17/20)** | **0.47%** | **98.75%** | 256.56 ms | 4.75% |

![ASR WER vs Semantic Similarity and Latency](docs/assets/asr_wer_vs_semantic_similarity.png)

![Whisper Model Distribution Histograms](docs/assets/lab8_cell41_fig1.png)

### 1.4 Domain-Stratified Error Breakdown & Downstream RAG Failure Taxonomy

| Domain | Utterances | `Base` Mean WER | `Distil` Mean WER | `Turbo` Mean WER | `Base` Sem. Sim. | `Distil` Sem. Sim. | `Turbo` Sem. Sim. | Primary Acoustic Vulnerability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Tech_NLP`** | 3 | 22.88% | 12.12% | **0.00%** | 86.26% | 91.50% | **100.00%** | Domain compound nouns (`"dense vectors"`, `"evaluating"`) |
| **`Food_Peru`** | 3 | 22.22% | 7.41% | **3.70%** | 79.45% | 94.75% | **97.32%** | Code-switched Spanish phonology (`"cevichería"`, `"leche de tigre"`) |
| **`History_War`** | 2 | 21.82% | 19.09% | **0.00%** | 91.97% | 93.41% | **100.00%** | Proper-noun homophones (`"axis powers"` $\to$ `"access powers"`) |
| **`Data_Eng`** | 2 | 13.64% | 13.64% | **0.00%** | 93.13% | 95.36% | **100.00%** | Technical initialisms (`"ssas and ssis"` $\to$ `"sas and sis"`) |
| **`Academic`** | 2 | 19.09% | 4.55% | **0.00%** | 82.01% | 96.45% | **100.00%** | Polysyllabic academic terms (`"graduate dissertation"`) |
| **`Support`** | 1 | 25.00% | **0.00%** | **0.00%** | 87.98% | **100.00%** | **100.00%** | Vowel laxing (`"reset"` $\to$ `"receipt"`) |
| **`Customer_Service`** | 1 | 21.43% | 7.14% | **0.00%** | 93.26% | 94.96% | **100.00%** | Date/number boundary (`"order 405"` $\to$ `"order 4-0-5"`) |
| **`Finance`** | 2 | 5.00% | 5.00% | **0.00%** | 94.68% | 94.68% | **100.00%** | Alphanumeric tax codes (`"w8ben form"` $\to$ `"w8b and form"`) |
| **`Business`** | 2 | 4.55% | **0.00%** | **12.12%** | 93.81% | **100.00%** | 94.69% | Acronym expansion (`"q3"` $\to$ `"quarter three"`) |
| **`Casual`** | 2 | **0.00%** | **0.00%** | **0.00%** | **100.00%** | **100.00%** | **100.00%** | High-frequency conversational priors |

Analysis of `data/asr_evaluation_results.csv` reveals three structural failure modes when piping ASR outputs into a Retrieval-Augmented Generation (RAG) or task-oriented NLU back-end:

1. **Entity Corruption Inducing Dense Retrieval Failure**:
   - In **`Tech_NLP_3`** (*"retrieval augmented generation relies on dense vectors"*), `Whisper-Base` transcribes *"dense vectors"* as **`"10 spectars"`** ($\text{WER} = 28.57\%, \text{Sim}_{\text{sem}} = 0.779$), while `Distil-Whisper` outputs **`"dense backdoors"`** ($\text{WER} = 14.29\%, \text{Sim}_{\text{sem}} = 0.873$). In a vector database query, substituting *"vectors"* with *"backdoors"* shifts the query embedding from information retrieval to cybersecurity, guaranteeing retrieval hallucination. `Whisper Large-v3-Turbo` resolves the utterance with **0.00% WER** ($\text{Sim}_{\text{sem}} = 1.000$).
   - In **`Data_Eng_2`** (*"migrating legacy pipelines from ssas and ssis to cloud"*), `Base` outputs **`"ssa's and ssis"`** and `Distil` outputs **`"sas and sis"`** ($\text{WER} = 18.18\%$), conflating Microsoft SQL Server Analysis/Integration Services (`SSAS`/`SSIS`) with SAS statistical software.
2. **Semantic Shift Altering User Intent**:
   - In **`Academic_1`** (*"working late on my master's graduate dissertation"*), `Whisper-Base` hallucinates **`"gravity certification"`** ($\text{WER} = 28.57\%, \text{Sim}_{\text{sem}} = 0.674$).
   - In **`Tech_NLP_2`** (*"evaluating the word error rate of speech recognition models"*), `Whisper-Base` transcribes *"evaluating"* as **`"evadiating"`** and `Distil-Whisper` outputs **`"invading"`** ($\text{WER} = 11.11\%$).
   - In **`Support_1`** (*"i need to reset my password immediately"*), `Whisper-Base` substitutes *"reset"* with **`"receipt"`** ($\text{WER} = 25.00\%$), routing an authentication ticket to billing.
3. **Catastrophic Failure on Regional & Code-Switched Entities**:
   - In **`Food_Peru_3`** (*"let's book a table at the peruvian cevichería tonight"*), `Whisper-Base` collapses to **`"sebi kerya"`** ($\text{WER} = 33.33\%, \text{Sim}_{\text{sem}} = 0.594$), `Distil-Whisper` outputs **`"cibiria"`** ($\text{WER} = 22.22\%$), and `Whisper Large-v3-Turbo` produces the near-exact ASCII transliteration **`"cevicheria"`** ($\text{WER} = 11.11\%$, $\text{CER} = 1.85\%$, $\text{Sim}_{\text{sem}} = 0.962$).
   - Interestingly, in **`Business_2`** (*"let's review the key performance indicators for q3"*), `Turbo` expands the spoken abbreviation `"q3"` into **`"quarter three"`** ($\text{WER} = 25.00\%$). While penalized by strict string-level WER, this normalization preserves 100% of the business intent.

---

## 2. Uncertainty-Aware Clarification Dialogue Manager (`clarification_dm.py`)

When ASR or user typing introduces noise into a task-oriented slot-filling system (Cambridge Restaurant Domain: `area`, `food`, `pricerange`), committing unverified slot guesses directly to the dialogue state causes **state pollution**—returning empty or wrong database results that require costly multi-turn user repairs.

### 2.1 Hierarchical Slot Normalization & Confidence Scoring

Our normalizer (`normalize_slot_verbose`) processes raw user inputs through a four-tier cascade over domain ontologies (`AREA_MAP`, `FOOD_MAP`, `PRICE_MAP`) paired with lexical domain hints (`AREA_HINTS`, `FOOD_HINTS`, `PRICE_HINTS`):

```mermaid
flowchart LR
    U["Raw User Utterance"] --> T1{"1. Exact Match<br/>in Ontology Map?"}
    T1 -->|"Yes"| S1["status = 'mapped'<br/>confidence = 1.00"]
    T1 -->|"No"| T2{"2. Substring Match<br/>in Utterance?"}
    T2 -->|"Yes"| S2["status = 'substring'<br/>confidence = 0.95"]
    T2 -->|"No"| T3{"3. Unigram/Bigram<br/>difflib Ratio ≥ 0.72?"}
    T3 -->|"Yes"| S3["status = 'fuzzy'<br/>confidence = r ∈ [0.72, 0.94)"]
    T3 -->|"No"| T4{"4. Domain Hint<br/>Mentioned?"}
    T4 -->|"Yes"| S4["status = 'unknown' (OOV)<br/>confidence = 0.0, mentioned = True"]
    T4 -->|"No"| S5["status = 'missing'<br/>confidence = 0.0, mentioned = False"]
```

### 2.2 Decision-Theoretic Grounding Policy

Let $c \in [0, 1]$ denote the normalized slot confidence. From a decision-theoretic Expected Utility ($EU$) perspective, the Dialogue Manager selects an action $a \in \{\text{Accept}, \text{Implicit}, \text{Explicit}, \text{Reject/Re-ask}\}$ maximizing:

$$\begin{aligned}
EU(\text{Accept} \mid c) &= c \cdot R_{\text{success}} + (1 - c) \cdot C_{\text{corrupt}} \\
EU(\text{Implicit} \mid c) &= c \cdot (R_{\text{success}} + C_{\text{imp}}) + (1 - c) \cdot \left(\tfrac{1}{2} R_{\text{success}} + \tfrac{1}{2} C_{\text{corrupt}} + C_{\text{imp}}\right) \\
EU(\text{Explicit} \mid c) &= R_{\text{success}} + C_{\text{exp}}
\end{aligned}$$

With calibrated thresholds $\tau_{\text{low}} = 0.84$ and $\tau_{\text{med}} = 0.94$, `decide_clarification` implements a strict priority hierarchy across `PRIORITY_SLOTS = ["food", "area", "pricerange"]`:

1. **Priority 1 — Mentioned Out-of-Ontology (`status == "unknown"`, `mentioned == True`)**:
   - Triggers **`explicit_clarification`** with `requires_options = True`, listing supported ontology values (e.g., *"I didn't catch the food type. Available options are: british, chinese, indian, italian, japanese, vegetarian. Which one would you like?"*).
2. **Priority 2 — Low-Confidence Fuzzy Match (`status == "fuzzy"`, $c < 0.84$)**:
   - Triggers **`explicit_clarification`** with `requires_options = False` (binary confirmation: *"I think you said centre for the area. Is that right?"*) and locks `state["pending_clarification"]`, blocking the guess from entering `state["constraints"]` until confirmed.
3. **Priority 3 — Medium-Confidence Fuzzy Match (`status == "fuzzy"`, $0.84 \le c < 0.94$)**:
   - Optimistically commits the slot via `update_state_from_parsed` and triggers **`implicit_clarification`**, embedding the assumed value into the next missing slot query (*"OK, in what area are you looking for a chinese restaurant?"*).
4. **Priority 4 — Unmentioned Required Slot (`status == "missing"`, `mentioned == False`)**:
   - Triggers a direct **`follow_up`** wh-question (*"What area are you looking for?"*).

![Clarification DM Policy Trade-off and Expected Utility](docs/assets/clarification_dm_policy_tradeoff.png)

### 2.3 Complete 12-Case Benchmark Evaluation (`Lab 9`)

| Case ID | Category | User Utterance | Parsed Slot & Status | Confidence $c$ | Expected Move | Predicted Move | Match | Generated System Response |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`M1`** | Missing | `"I want Chinese food."` | `food="chinese"` (`substring`) | `0.9500` | `follow_up` | `follow_up` | Pass | *"What area are you looking for?"* |
| **`M2`** | Missing | `"I want a cheap place."` | `pricerange="cheap"` (`substring`) | `0.9500` | `follow_up` | `follow_up` | Pass | *"What kind of food would you like?"* |
| **`M3`** | Missing | `"Find me somewhere in the centre."` | `area="centre"` (`substring`) | `0.9500` | `follow_up` | `follow_up` | Pass | *"What kind of food would you like?"* |
| **`A1`** | Ambiguous | `"I want chainese food."` | `food="chinese"` (`fuzzy`) | `0.9333` | `implicit` | `implicit` | Pass | *"OK, in what area are you looking for a chinese restaurant?"* |
| **`A2`** | Ambiguous | `"Find me something in the senter."` | `area="centre"` (`fuzzy`, `"senter."`) | `0.7692` | `implicit` | `explicit` | **Fail** | *"I think you said centre for the area. Is that right?"* |
| **`O1`** | OOV | `"I want asian food."` | `food=None` (`unknown`, mentioned) | `0.0000` | `explicit` | `explicit` | Pass | *"I didn't catch the food type. Available options are: ... Which one would you like?"* |
| **`O2`** | OOV | `"Find me a midscale place."` | `pricerange=None` (`unknown`, mentioned) | `0.0000` | `explicit` | `explicit` | Pass | *"I didn't catch the price range. Available options are: cheap, expensive, moderate..."* |
| **`O3`** | OOV | `"I'm looking for oriental food in the west."` | `food=None` (`unknown`), `area="west"` (`0.95`) | `0.0000` | `explicit` | `explicit` | Pass | *"I didn't catch the food type. Available options are: ... Which one would you like?"* |
| **`N1`** | Noisy ASR | `"i want a cheap chainese place"` | `food="chinese"` (`fuzzy`), `price="cheap"` (`0.95`) | `0.9333` | `implicit` | `implicit` | Pass | *"OK, in what area are you looking for a chinese restaurant?"* |
| **`N2`** | Noisy ASR | `"modrate japanees food"` | `food="japanese"` (`0.875`), `price="moderate"` (`0.933`) | `0.8750` | `implicit` | `implicit` | Pass | *"OK, in what area are you looking for a japanese restaurant?"* |
| **`N3`** | Noisy ASR | `"i want a restarant in the senter"` | `area="centre"` (`fuzzy`, `"senter"`) | `0.8333` | `explicit` | `explicit` | Pass | *"I think you said centre for the area. Is that right?"* |
| **`N4`** | Noisy ASR | `"book for to people at seven"` | all slots `missing` (`mentioned=False`) | `0.0000` | `follow_up` | `follow_up` | Pass | *"What kind of food would you like?"* |

### 2.4 Proof of the Token-Level Threshold Contradiction (`A2` vs. `N3`)

Why does the baseline policy achieve **91.67% (11/12)** and fail specifically on `A2`? Inspecting the candidate n-grams reveals a subtle interaction between **surface punctuation** and **utterance-level vs. token-level uncertainty**:

1. **Punctuation-Induced Confidence Penalization in `A2`**:
   - In `A2` (`"Find me something in the senter."`), whitespace splitting `.split()` leaves the trailing period attached to `"senter."` (7 characters). Against `"center"` (6 characters), `difflib.SequenceMatcher` computes:
     $$c(\text{"senter."}, \text{"center"}) = \frac{2 \times M}{T} = \frac{2 \times 5}{7 + 6} = \frac{10}{13} \approx 0.7692 < \tau_{\text{low}} = 0.84$$
2. **The `A2` vs. `N3` Structural Contradiction**:
   - If we strip punctuation prior to candidate generation (`strip_punctuation=True`), `"senter"` in `A2` achieves:
     $$c(\text{"senter"}, \text{"center"}) = \frac{2 \times 5}{6 + 6} = \frac{10}{12} \approx 0.8333$$
   - If we then relax $\tau_{\text{low}}$ from `0.84` to `0.83`, **`A2` flips to `Pass` (`implicit`)**, **but `N3` (`"i want a restarant in the senter"`) simultaneously flips to `Fail` (`implicit` instead of expected `explicit`)**!
   - **Theoretical Insight**: Both `A2` and `N3` contain the exact same corrupted slot token `"senter"` ($c = 0.8333$). However, `N3` has **global utterance corruption** (`"restarant"` + missing punctuation), whereas `A2` is a clean sentence with a single phonetic typo. A purely **local, token-level slot normalizer** assigns identical confidence $c = 0.8333$ to the `area` slot in both utterances, making it mathematically impossible for any scalar threshold $\tau_{\text{low}} \in \mathbb{R}$ to separate `A2` from `N3` without conditioning on a **global sentence-level acoustic/syntactic coherence factor** $\gamma_{\text{utt}} \in (0, 1]$.

---

## 3. Multi-Party Conversation & Next-Speaker Prediction (`multiparty.py`)

In dyadic dialogue, turn-taking reduces to binary floor transfer between user and system. In **multi-party meetings**, any of $K=4$ participants $\{A, B, C, D\}$ may hold the floor, self-select, or be addressed next.

### 3.1 AMI Meeting Corpus Aggregation & Sliding-Window Construction

From the **AMI Meeting Corpus** word-level XML annotations (`986,960` words across `171` meetings), we construct continuous speaker turns via `build_turns(words_df, max_gap=0.5)`:
- Consecutive words spoken by the same participant $s_t$ with inter-word silence gap $\Delta t = t_{\text{start}}^{(i)} - t_{\text{end}}^{(i-1)} \le 0.5\text{ s}$ are merged into a single utterance turn, yielding **212,312 turns**.
- Filtering to meetings with all 4 canonical roles $\{A, B, C, D\}$ active retains **147 meetings (197,482 turns)**.
- Applying a sliding window of $H = 8$ historical turns (`build_multiparty_dataset`) produces **196,159 supervised instances** split by meeting ID into **Train (`154,628`)**, **Dev (`18,616`)**, and **Test (`22,915`)** with a word vocabulary of **10,444 tokens** (`MIN_FREQ = 2`, `MAX_LEN = 40`).

### 3.2 Hierarchical `NextSpeakerBiLSTMModel` Architecture

Given the 8-turn history $\mathcal{H}_t = \{(u_{t-8}, s_{t-8}), \dots, (u_{t-1}, s_{t-1})\}$ and current turn $(u_t, s_t)$:

1. **Word-Level Utterance BiLSTM (`UtteranceBiLSTMEncoder`)**: Each utterance $u_k$ of length $\ell_k$ is embedded ($d_w = 128$) and encoded via `pack_padded_sequence` through a 1-layer Bidirectional LSTM ($d_h = 128$ per direction), yielding $\mathbf{v}_k = [\overrightarrow{\mathbf{h}}_k; \overleftarrow{\mathbf{h}}_k] \in \mathbb{R}^{256}$.
2. **Historical Speaker Identity Fusion**: Each historical speaker $s_k \in \{0, 1, 2, 3\}$ is mapped to a learned embedding $\mathbf{e}_{s_k} \in \mathbb{R}^{16}$ and concatenated: $\tilde{\mathbf{v}}_k = [\mathbf{v}_k; \mathbf{e}_{s_k}] \in \mathbb{R}^{272}$ (padded with $\mathbf{0}_{16}$ at step $t$).
3. **Turn-Sequence Context BiLSTM**: The sequence $(\tilde{\mathbf{v}}_{t-8}, \dots, \tilde{\mathbf{v}}_t) \in \mathbb{R}^{9 \times 272}$ is processed by a context-level BiLSTM ($d_c = 128$ per direction) to produce the discourse state $\mathbf{c}_t \in \mathbb{R}^{256}$.
4. **Current Speaker Conditioning & Classification**: Concatenating the current speaker embedding $\mathbf{e}_{s_t} \in \mathbb{R}^{16}$ yields $\mathbf{z}_t = [\mathbf{c}_t; \mathbf{e}_{s_t}] \in \mathbb{R}^{272}$, which passes through a two-layer MLP ($272 \to 128 \to 4$) optimized with AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$, early stopping patience $5$).

### 3.3 Quantitative Ablation Study on the AMI Test Set

| Model Configuration | `use_history` ($H$) | `use_speaker_info` | Dev Peak Macro-F1 | Test Accuracy (%) | Test Macro-F1 (%) | Test Precision (%) | Test Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Hierarchical BiLSTM** | $H = 8$ turns | `True` (Hist + Cur) | **71.99%** (Ep 10) | **70.72%** | **70.48%** | **70.54%** | **70.43%** |
| **Ablation 1: No Dialogue History** | $H = 0$ turns | `True` (Cur only) | 27.72% | 33.42% | 27.26% | — | — |
| **Ablation 2: No Speaker Identity** | $H = 8$ turns | `False` (Text only) | 25.79% | 27.36% | 27.26% | — | — |
| *Uniform 4-Speaker Baseline* | — | — | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% |

![AMI Multi-Party Ablation Validation F1 Curves](docs/assets/lab10_cell33_fig1.png)

![AMI Multi-Party Normalized Confusion Matrix](docs/assets/lab10_cell43_fig2.png)

### 3.4 Turn-Boundary Error Taxonomy & Discourse Dynamics

1. **Why Removing Speaker Identity Collapses to Random Guessing (`27.36% Acc`)**:
   - Multi-party meetings are governed by **floor-holding inertia** (a speaker continuing across short pauses) and **dyadic ping-pong reciprocity** ($A \leftrightarrow B$ exchanges). Without speaker IDs (`use_speaker_info=False`), the model sees what was said (`"Yeah"`, `"Okay"`) but does not know *who* holds the floor or *who* was previously speaking, collapsing to the 25% uniform prior.
2. **Why Removing History Destroys Macro-F1 (`27.26% F1`)**:
   - Without the 8-turn context (`history=0`), when speaker $B$ utters a 1-word backchannel (`"Yeah."`), the model cannot identify which participant ($A, C,$ or $D$) held the floor prior to the backchannel and is about to resume speaking.
3. **False Switches vs. Missed Switches**:
   - Evaluating all **18,035 test transitions**, the Full Model commits **734 False Switches** (predicting a speaker change when the current speaker actually continues) versus only **204 Missed Switches** (predicting floor-holding when a turn switch actually occurs) — a **3.6$\times$ asymmetry**:
     - *False Switches (734)* occur when a speaker pauses for $>0.5\text{ s}$ after a syntactically complete clause (`"Okay , let's move on ."`), which `build_turns` segments into two turns; the model anticipates an addressee response before the speaker continues.
     - *Missed Switches (204)* occur when a speaker trails off mid-clause (`"Um ."`, `"So that's uh"`, `"um button"`); because the utterance appears syntactically unfinished, the BiLSTM predicts floor retention even though another participant interrupts to seize the floor.

---

## 4. Multimodal Referring Expression Comprehension & Grounding (`multimodal_grounding.py`)

In **GuessWhat?!** (De Vries et al., 2017), three agents collaborate over an MS-COCO image containing $K$ candidate objects (`3 <= K <= 6`, bounding boxes $\ge 1500\text{ px}^2$) to isolate a secret target object $c^* \in \{0, \dots, K-1\}$ within $T = 5$ binary QA turns:
1. **Questioner**: Sees the image and candidate list (`[idx] category {"bbox_2d": [xmin, ymin, xmax, ymax]}` normalized to Qwen-VL's $[0, 999]$ grid) and asks Yes/No questions.
2. **Oracle**: Knows the secret target index $[c^*]$ and answers strictly `"Yes"`, `"No"`, or `"N/A"`.
3. **Selector**: Reviews the image, candidate list, and full 5-turn QA transcript to output the predicted index `[idx]`.

![GuessWhat Candidate Annotations Example 1](docs/assets/lab11_cell9_fig2.png)

![GuessWhat Candidate Annotations Example 2](docs/assets/lab11_cell9_fig4.png)

### 4.1 Bifurcated Inference & Prompting Grid Search (`Qwen/Qwen3.5-0.8B`)

Using a compact 0.8B-parameter Vision-Language Model (`Qwen/Qwen3.5-0.8B`) across all three roles exposes severe **role confusion** (the Questioner answering its own questions or describing the image) and **echolalia** (the Selector copying few-shot demonstration indices). We engineered **Bifurcated Inference**:
- **Stochastic Sampling (`do_sample=True`, `repetition_penalty=1.2`)** for the **Questioner** to prevent repetitive question loops across the 5 turns.
- **Deterministic Greedy Decoding (`do_sample=False`, `max_new_tokens=5`)** for the **Oracle** and **Selector** (with logit prefill `"["`) to enforce strict output formatting.

| Configuration | Questioner Prompt Strategy | Questioner Prefill | Temp $\tau$ | Oracle Prompting | Run 1 Acc (%) | Run 2 Acc (%) | Oracle Strict Valid (%) | Oracle Recoverable (%) | Eval Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Config A`** | High-Constraint Negative Rules | `"Is it "` | `0.7` | 1-Shot | 20.0% | 20.0% | 68.0% | 95.0% | 717s |
| **`Config B`** | **High-Constraint Negative Rules** | **`"Is it "`** | **`0.7`** | **0-Shot** | **45.0%** | **50.0%** | **91.0%** | **100.0%** | **619s** |
| **`Config C`** | **Low-Constraint Positive Example** | **None (`""`)** | **`0.5`** | **0-Shot / 1-Shot** | **25.0%** | **35.0%** | **76.0%** | **100.0%** | **784s** |
| **`Config D`** | High-Constraint Negative Rules | `"Is it "` | `0.3` | 1-Shot | 15.0% | 15.0% | 65.0% | 94.0% | 684s |
| **`Config B + LoRA`** | High-Constraint Negative Rules | `"Is it "` | `0.7` | **LoRA ($r=32$)** | **45.0%** | **45.0%** | **100.0%** | **100.0%** | 5865s |
| **`Config C + LoRA`** | Low-Constraint Positive Example | None (`""`) | `0.5` | **LoRA ($r=32$)** | **30.0%** | **30.0%** | **100.0%** | **100.0%** | 2562s |

![Multimodal Grounding Accuracy and Oracle Response Shift](docs/assets/multimodal_grounding_accuracy.png)

### 4.2 Emergent Shortcut Learning (`Config B`) vs. True Visual Grounding (`Config C`)

Comparing the dialogue transcripts of `Config B` (50.0% accuracy) and `Config C` (35.0% accuracy) uncovers a fundamental phenomenon in multi-agent VLM communication:

- **`Config B` (Emergent Index-Enumerating Shortcut)**:
  - Because `Config B` forces every question to start with `"Is it "` and supplies numbered candidates `[0]..[5]` in the system prompt, the 0.8B Questioner discovers a **non-visual game-theoretic shortcut**:
    > `Turn 1: Is it 1st candidate? -> Yes`
    > `Turn 2: Is it 2nd candidate? -> No`
    > `Turn 3: Is it 3rd candidate? -> No`
  - While this maximizes Selector accuracy (50.0%, nearly $2\times$ the ~25% random baseline), it bypasses visual feature extraction entirely—and triggers **Oracle Sycophancy** (the zero-shot 0.8B Oracle answers `"Yes"` to 48% of queries, often agreeing to multiple candidate indices within the same game).
- **`Config C` (Genuine Visual-Spatial Feature Interrogation)**:
  - Removing the `"Is it "` prefill and lowering temperature to $\tau = 0.5$ enables genuine multimodal spatial reasoning:
    > `Turn 1: Is there any other vehicle visible besides the one with red lights on its top right side? -> No`
    > `Turn 2: Is this car parked next to another car or truck? -> Yes`
    > `Turn 3: Does the person wear glasses? -> No`
  - However, without logit prefill, the 0.8B Questioner occasionally produces **syntactic blending** (`"Is it Who is holding these rackets?"`), which causes the zero-shot Oracle to output unparseable explanations (`"None"`, dropping strict validity to **76.0%**).

### 4.3 LoRA Fine-Tuning of the Oracle & Asymmetric Multi-Agent Breakdown

To eliminate Oracle formatting failures and sycophancy, we fine-tuned `Qwen/Qwen3.5-0.8B` as a specialized Oracle using **Low-Rank Adaptation (LoRA)** (`r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, target modules `["q_proj", "v_proj", "k_proj", "o_proj"]`, `fp16`, batch size 1 with 16-step gradient accumulation, 3 epochs over 392 training / 54 validation GuessWhat?! QA pairs):

![Qwen3.5-0.8B LoRA Oracle Training and Validation Loss](docs/assets/lab11_cell64_fig5.png)

- **100.00% Strict Format Compliance**: LoRA fine-tuning reduces validation loss smoothly from `1.72` to `0.51` and drives `strict_valid_rate` to **100.00%** across both `Config B` and `Config C` (zero `"None"` failures).
- **The Asymmetric Multi-Agent Fine-Tuning Breakdown**:
  - Despite perfect Oracle formatting, end-to-end game accuracy shifts slightly downward (`50% -> 45%` in `Config B`; `35% -> 30%` in `Config C`).
  - **Root Cause**: Because ground-truth GuessWhat?! training data contains predominantly `"No"` responses to early narrowing questions, the LoRA-tuned Oracle shifts from **Zero-Shot Sycophancy** (48% `"Yes"` / 43% `"No"` in `Config B`) to **Hyper-Conservative Rejection** (**86% `"No"` / 14% `"Yes"` in `Config B**; **74% `"No"` / 26% `"Yes"` in `Config C`**).
  - Starved of positive `"Yes"` confirmations across 3+ consecutive turns, the **un-tuned 0.8B Questioner** suffers **autoregressive context collapse** in Turn 4–5, leaking raw vision special tokens (`"Is it <image> <image> ..."`), which deprives the Selector of usable evidence. This proves that **unilaterally fine-tuning a single agent in a multi-agent VLM loop breaks co-adapted conversational priors**; Questioner and Oracle must be co-trained or aligned via joint reinforcement learning.

---

## 5. Neural Dialogue Foundations: Act Tagging (`dialogue_act.py`) & Seq2Seq Generation (`seq2seq_generator.py`)

![Summary of Dialogue Act, Seq2Seq, and Multi-Party Ablations](docs/assets/dialogue_models_ablation_summary.png)

### 5.1 Switchboard Dialogue Act Tagging (`Lab 2`)

We model utterance-level communicative intent across **223,606 utterances** (`192,390` train / `31,216` test, vocabulary `43,732`) from the **Switchboard Dialogue Act Corpus (`SwDA`)**, simplified via `damsl_act_tag` into **43 canonical DAMSL tags**.

![Switchboard 43-Tag Zipfian Distribution](docs/assets/lab2_cell52_fig1.png)

The corpus exhibits extreme **Zipfian class imbalance**: Statement-non-opinion (`sd`, **32.6%**) and Acknowledge/Backchannel (`b`, **16.7%**) account for nearly half of all turns, whereas critical conversational repair acts—**Signal-non-understanding (`br`, 0.13%, 288 total instances)** and **Summarize/Reformulate (`bf`, 0.42%, 918 total instances)**—occupy the extreme tail.

| Model Architecture | Loss Weighting | Epochs | Test Loss | Overall Test Acc (43 Tags) | `br` (Non-Understanding, $N=53$) | `bf` (Reformulate, $N=250$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: 2-Layer BiLSTM (Mean Pool)** | Unweighted CE | 5 | 1.0683 | 67.96% | 0.00% (0/53) | 0.00% (0/250) |
| **Model 2a: 2-Layer BiLSTM (Balanced)** | Inverse-Freq $w_c = \frac{N}{C \cdot n_c}$ | 5 | 2.2324 | 34.83% | **54.72% (29/53)** | 2.80% (7/250) |
| **Model 2b: 2-Layer BiLSTM (Balanced)** | Inverse-Freq $w_c = \frac{N}{C \cdot n_c}$ | 20 | 1.9764 | 37.01% | **54.72% (29/53)** | **12.40% (31/250)** |
| **Model 3: Dual-Path `CNN_BiLSTM`** | Unweighted CE | 5 | **0.9414** | **70.63%** | **37.74% (20/53)** | 0.00% (0/250) |

- **Why `CNN_BiLSTM` Recovers `br` Without Sacrificing Majority Accuracy**:
  - Inverse-frequency weighting (`Model 2`) boosts `br` recall to **54.72%**, but inflates gradients on noisy tail classes by $>250\times$, collapsing overall accuracy from **67.96% $\to$ 34.83%**.
  - Our **`CNN_BiLSTM` (`Model 3`)** pairs parallel 2D Convolutional filters ($k \in \{3, 4, 5\}$, 64 filters each + BatchNorm + MaxPool) with a 2-layer BiLSTM context encoder. Because `br` repairs rely on short, invariant lexical n-grams (*"Excuse me?"*, *"What did you say?"*, *"Huh?"*), the spatial CNN path isolates `br` patterns (**37.74% recall**) while boosting overall 43-class accuracy to **70.63%**. Conversely, `bf` (*Summarize/Reformulate*) requires cross-turn semantic comparison with the partner's prior utterance, which single-utterance n-grams cannot capture without explicit multi-turn context.

### 5.2 Bahdanau Attention Seq2Seq & Residual `Add & Norm` Decoder (`Lab 3`)

We implement an neural sequence-to-sequence response generator (`seq2seq_generator.py`) initialized with frozen **50-D GloVe embeddings**, a **Bidirectional GRU Encoder** ($256 \times 2 = 512$ units), **Additive Bahdanau Attention**:

$$e_{t,i} = \mathbf{v}_a^\top \tanh(\mathbf{W}_1 \mathbf{s}_{t-1} + \mathbf{W}_2 \mathbf{h}_i), \qquad \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})}, \qquad \mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{t,i} \mathbf{h}_i$$

and evaluate across two distinct conversational regimes:
1. **Open-Domain Social Chat (`Cornell Movie-Dialogs`, 89,924 pairs, vocab 35,495, 60 epochs)**: Loss decreases from `5.30 -> 2.45`, exhibiting diffuse attention weights across conversational fillers (`"you"`, `"know"`) and generic safe responses (`"i do not know"`).
2. **Knowledge-Grounded Factual QA (`HybridDialogue`, 4,359 Wikipedia-grounded pairs, vocab 6,605, 50 epochs)**:

| Decoder Architecture & Hyperparameters | Layers ($L$) | Dropout ($p$) | Epoch 1 Loss | Epoch 50 Loss | Qualitative Generation & Attention Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config 1: Baseline (`Decoder`)** | $L = 1$ | $p = 0.2$ | 6.3859 | 1.2123 | Fast convergence; sharp diagonal entity alignment (`"clement"`, `"1992"`), mild late-epoch memorization |
| **Config 2: High Dropout (`Decoder`)** | $L = 1$ | $p = 0.5$ | 6.3530 | 1.3000 | Higher numerical loss due to stochastic regularization, but **superior factual recall** (`"four years"` for presidential term) |
| **Config 3: Deeper Stacked (`Decoder`)** | $L = 2$ | $p = 0.2$ | 6.4200 | 2.0400 | **Gradient dilution & n-gram repetition loops** (`"is the the the..."`); attention signal attenuated across second GRU |
| **Config 4: Deeper + High Drop (`Decoder`)** | $L = 2$ | $p = 0.5$ | 6.4500 | 2.0100 | Severe underfitting; compound recurrent depth + heavy dropout starves parameter updates on 4,359 pairs |
| **Proposed `DecoderAdvanced` (`Add & Norm`)** | **$L = 2$ (Residual)** | **$p = 0.3$** | 5.7975 | **0.1060** | Smooth optimization; eliminates repetition loops while preserving multi-layer representational capacity |
| **Proposed `DecoderAdvanced` (`Add & Norm`)** | **$L = 2$ (Residual)** | **$p = 0.2$** | **5.7805** | **0.0830** | **15$\times$ loss reduction vs. 1-layer baseline (`1.2123`)** and **25$\times$ vs. naive 2-layer (`2.0400`)** |

![Seq2Seq 4-Configuration Ablation Training Curves](docs/assets/lab3_cell80_fig27.png)

![Seq2Seq Residual Add and Norm Decoder Convergence](docs/assets/lab3_cell83_fig28.png)

#### Why `DecoderAdvanced` (`Add & Norm`) Solves Stacked Recurrent Degradation

In a naive 2-layer decoder (`Config 3`), the Bahdanau context vector $\mathbf{c}_t$ only enters `GRU_1`; `GRU_2` receives only the intermediate hidden state $\mathbf{o}_t^{(1)}$, causing vanishing gradients and decoupling the output projection $\mathbf{W}_{\text{fc}}$ from direct attention alignments. In `DecoderAdvanced`, we introduce a **Residual Skip Connection** and **Layer Normalization** prior to vocabulary projection:

$$\mathbf{o}_t^{(1)}, \mathbf{h}_t^{(1)} = \text{GRU}_1([\mathbf{c}_t; \mathbf{e}(y_{t-1})], \mathbf{h}_{t-1}^{(1)}), \qquad \mathbf{o}_t^{(2)}, \mathbf{h}_t^{(2)} = \text{GRU}_2(\text{Dropout}(\mathbf{o}_t^{(1)}), \mathbf{h}_t^{(1)})$$

$$\mathbf{z}_t = \text{LayerNorm}\!\left(\mathbf{o}_t^{(1)} + \mathbf{o}_t^{(2)}\right), \qquad P(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_{\text{fc}} \mathbf{z}_t)$$

This creates a direct gradient highway ($\frac{\partial \mathbf{z}_t}{\partial \mathbf{o}_t^{(1)}} = \mathbf{I} + \frac{\partial \mathbf{o}_t^{(2)}}{\partial \mathbf{o}_t^{(1)}}$) directly to the attention-conditioned first layer, driving Epoch 50 cross-entropy loss down to **0.0830**.

| Attention Alignment Example A (`HybridDialogue`) | Attention Alignment Example B (`HybridDialogue`) |
| :---: | :---: |
| ![Bahdanau Attention Heatmap 1](docs/assets/lab3_cell71_fig25.png) | ![Bahdanau Attention Heatmap 2](docs/assets/lab3_cell72_fig26.png) |

---

## Repository Structure

```text
multimodal-spoken-dialogue-systems/
├── .github/workflows/
│   └── ci.yml                          # GitHub Actions CI (pytest + CLI benchmark verification)
├── data/
│   └── asr_evaluation_results.csv      # 20-utterance multi-domain accented ASR benchmark dataset
├── docs/assets/                        # Publication-grade summary figures & extracted lab plots
│   ├── asr_wer_vs_semantic_similarity.png
│   ├── clarification_dm_policy_tradeoff.png
│   ├── multimodal_grounding_accuracy.png
│   ├── dialogue_models_ablation_summary.png
│   ├── lab2_cell52_fig1.png            # Switchboard 43-tag Zipfian class distribution
│   ├── lab3_cell71_fig25.png           # Bahdanau attention alignment heatmap (Example 1)
│   ├── lab3_cell72_fig26.png           # Bahdanau attention alignment heatmap (Example 2)
│   ├── lab3_cell80_fig27.png           # Seq2Seq 4-configuration ablation training curves
│   ├── lab3_cell83_fig28.png           # Residual Add & Norm decoder training curves
│   ├── lab8_cell41_fig1.png            # Whisper Base/Distil/Turbo WER & latency histograms
│   ├── lab10_cell33_fig1.png           # AMI Multi-Party ablation validation F1 trajectories
│   ├── lab10_cell43_fig2.png           # AMI Multi-Party 4-speaker normalized confusion matrix
│   ├── lab11_cell9_fig2.png            # GuessWhat?! COCO candidate bounding box annotations (Scene 1)
│   ├── lab11_cell9_fig4.png            # GuessWhat?! COCO candidate bounding box annotations (Scene 2)
│   └── lab11_cell64_fig5.png           # Qwen3.5-0.8B LoRA Oracle training & validation loss
├── scripts/
│   ├── evaluate_spoken_pipeline.py     # CLI: Spoken ASR benchmark evaluation & plot generator
│   └── run_dialogue_benchmarks.py      # CLI: Clarification DM, Multi-Party, VLM & Seq2Seq benchmarks
├── src/dialogue_systems/
│   ├── __init__.py                     # Unified package exports
│   ├── spoken_asr.py                   # Levenshtein WER/CER, semantic similarity & Whisper analysis
│   ├── clarification_dm.py             # Hierarchical slot normalizer, EU model & grounding policy
│   ├── multiparty.py                   # AMI turn aggregation, UtteranceBiLSTM & NextSpeakerBiLSTMModel
│   ├── multimodal_grounding.py         # Qwen [0, 999] bbox normalizer, GuessWhat?! prompts & fusion
│   ├── dialogue_act.py                 # Switchboard DAMSL mapper, BiLSTMModel & Dual-Path CNN_BiLSTM
│   └── seq2seq_generator.py            # BahdanauAttention, BiGRU Encoder, Decoder & DecoderAdvanced
├── tests/
│   └── test_dialogue_systems.py        # Comprehensive pytest suite covering all 6 modules
├── LICENSE                             # MIT License
├── pyproject.toml                      # PEP 621 package configuration
└── requirements.txt                    # Reproducible runtime & test dependencies
```

---

## Installation & Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Ceviche98/multimodal-spoken-dialogue-systems.git
cd multimodal-spoken-dialogue-systems
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Unit Test Suite

```bash
pytest -v
```

### 3. Reproduce All Benchmarks & Figures

```bash
# 1. Evaluate the 20-utterance accented ASR benchmark (Whisper Base vs Distil-Large-v3 vs Turbo)
python scripts/evaluate_spoken_pipeline.py

# 2. Run Clarification DM (12 cases), Dialogue Act, Seq2Seq, Multi-Party AMI, and GuessWhat?! VLM benchmarks
python scripts/run_dialogue_benchmarks.py
```

---

## License

Released under the [MIT License](LICENSE).
