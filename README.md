# Taiji Architecture: From QKV Deconstruction to Pure Physical Emergence

## Introduction

Modern large language models are dominated by probability matrices. While the Transformer architecture has achieved remarkable success, its reliance on Softmax (a zero-sum probability game) and high-dimensional feature mixing (QKV) introduces fundamental limitations: numerical instability in long contexts, heavy KV Cache memory footprints, and a structural inability to express "Absolute Silence" (zero interference).

**The Taiji Architecture** is a fundamental reconstruction of the language model's bottom layer. Based on first principles of geometry and physics, we replace probability distributions with a pure physical dynamics engine. 

This repository documents the complete research journey across four phases, culminating in **Phase 4 (V20)**—a pure geometric emergence engine that eliminates both the Softmax mechanism and the final linear output layer.

---

## Research Roadmap (The Evolutionary Journey)

### Phase 1: Single-Sphere Geometric Evolution (`/phase1_single_sphere`)
*   **Goal**: Abolish QKV projections and multi-head mechanisms. Decompose semantic vectors purely into **Direction (Phase/Position)** and **Scalar (Energy/Value)**.
*   **Mechanism**: Attention is calculated via pure geometric phase differences rather than Softmax.
*   **Result**: KV Cache usage reduced by ~36%.

### Phase 2: Mass-Center Integration (`/phase2_center_of_mass`)
*   **Goal**: Replace Softmax's zero-sum probability framework to physically express "Absolute Silence".
*   **Mechanism**: Introduced Newtonian mechanics: **Mass-Center Integration**. The current token dynamically merges into the global historical center of mass.
*   **Result**: SST-2 hit 72.71%. Exact Zero-Interference (0.000000) achieved. However, it **failed miserably at Copy Tasks** because linear averaging destroys precise micro-memory.

### Phase 3: Macro-Micro Unified Field (`/phase3_unified_field`)
*   **Goal**: Balance macro-context and micro-memory.
*   **Mechanism**: Introduced a multi-order force field (1st-order macro-force for global context, 3rd-order micro-force for sharp local resonance). It worked, but still relied on a final `nn.Linear` layer to output probabilities—an architectural compromise.

### Phase 4: Pure Physical Emergence - V20 (`/phase4_physical_emergence`)
*   **Goal**: Complete architectural purity. Remove the final linear layer entirely.
*   **Mechanism**: The next token is physically "emerged" via geometric distance matching (finding the closest vocabulary vector on the 128-dim sphere). No QKV, No Softmax, No Linear Head.

---

## Core Experimental Results (Phase 4 / V20)

Tested on a 2-layer, 128-dim model without pre-training:

*   **SST-2 Sentiment Classification**: **72.36%** (Standard Transformer baseline is 71.10%).
*   **Copy Task (Micro-memory)**: Loss drops to **0.0042**.
*   **Absolute Silence (Zero-Interference)**: By mathematically constructing an isolated token strictly orthogonal (via SVD) to ALL past tokens, the force field vanishes. The state transition step is exactly **0.000000**. Softmax fundamentally cannot represent this absolute "Nothingness".

### Mechanism Comparison

| Dimension | Traditional Transformer | Taiji V20 Architecture |
| :--- | :--- | :--- |
| **Projection Matrices** | Relies on independent \(W_q, W_k, W_v\) | No projection matrices; directly decomposes token vectors |
| **Feature Aggregation** | High-dimensional vector mixing (Feature Pollution) | Pure scalar energy / Force field aggregation |
| **Inference Cache** | Requires caching massive Key and Value matrices | Only requires caching direction and scalar energy |
| **Output Layer** | Final `nn.Linear` + Softmax | Pure geometric distance matching |
| **"Unknown" Expression** | Impossible (Forced probability distribution) | **Absolute Silence (0.000000)** |

### What we inherited from Transformer
We did not rebuild everything from scratch. The foundations remain standard:
*   **Embeddings:** Token and positional embeddings map discrete text into the physical sphere.
*   **MLP + GELU:** Acts as the non-linear manifold alignment layer after the physical forces have been applied.
*   **Residual Connections:** Ensures the "ontological anchor" (the original semantic base) is never lost during violent force interactions.
*   **Bias = False:** Crucially, we removed all biases in MLPs to guarantee true `F(0) = 0` zero-interference.
---

## Honest Ablation & Limitations (The Physics Boundaries)

Scientific truth requires acknowledging boundaries. We ran rigorous ablations (see `/phase4_physical_emergence/ablation/`):

1.  **Macro vs. Micro on SST-2**: Using *only* the 1st-order macro-force achieves **73.51%** on SST-2, slightly outperforming the unified field (72.36%). This proves that sentiment classification requires smooth global context; the sharp 3rd-order micro-force introduces slight local noise to the global smoothing.
2.  **Micro-force on Copy Task**: Without the 3rd-order force, Copy Loss stalls at 0.0102. With it, Loss drops to 0.0042. The micro-force is strictly specialized for exact local memory.
3.  **Information-Theoretic Limit**: In a highly noisy, random Copy task (30% injected random tokens), both macro-only and unified models collapse (Loss ~2.5). This confirms that pure physical force fields cannot predict absolute random chaos.

---

## Repository Structure

```text
📦 Taiji-Architecture
 ┣ 📂 phase1_single_sphere/       # Phase 1: Geometric evolution & QKV abolition
 ┃ ┣ 📜 model.py
 ┃ ┣ 📜 main.py
 ┃ ┣ 📜 README.md
 ┃ ┣ 📜 Phase1_Technical_Report_EN.pdf
 ┃ ┣ 📜 requirements.txt
 ┃ ┗ 📂 assets/
 ┣ 📂 phase2_center_of_mass/      # Phase 2: Mass-center dynamics & Absolute Silence
 ┃ ┣ 📜 model.py
 ┃ ┣ 📜 main.py
 ┃ ┣ 📜 README.md
 ┃ ┣ 📜 Phase2_Technical_Report_EN.pdf
 ┃ ┣ 📜 requirements.txt
 ┃ ┗ 📂 assets/
 ┣ 📂 phase3_unified_field/       # Phase 3: Macro-Micro Unified Field (V19)
 ┃ ┣ 📜 model.py
 ┃ ┣ 📜 main.py
 ┃ ┣ 📜 README.md
 ┃ ┣ 📜 requirements.txt
 ┃ ┗ 📂 assets/
 ┣ 📂 phase4_physical_emergence/  # Phase 4: Pure Physical Emergence (V20)
 ┃ ┣ 📜 model.py
 ┃ ┣ 📜 main.py
 ┃ ┣ 📜 README.md
 ┃ ┣ 📜 requirements.txt
 ┃ ┣ 📂 ablation/
 ┃ ┗ 📂 assets/
 ┗ 📜 README.md                   # This document
```

---

## Code Execution

To reproduce the V20 results (Phase 4):

```bash
# 1. Clone the repository
git clone https://github.com/Wangzhifang199127/Taiji-Single-Sphere.git
cd Taiji-Single-Sphere

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Phase 4 (V20) training, evaluation, and physical diagnostics
cd phase4_physical_emergence
python main.py
```

---

## Future Work

The current prototype is extremely small (2 layers, 128 dims). **Scaling laws remain completely untested.** Our next research frontier focuses on a Universal Capture Mechanism (e.g., Wave-Particle Duality) to further optimize long-context deployment and numerical stability at scale.

---

## Citation

```bibtex
@misc{taiji_single_sphere_2024,
  author = {Wang Zhifang},
  title = {Taiji Architecture: From QKV Deconstruction to Pure Physical Emergence},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Wangzhifang199127/Taiji-Single-Sphere}}
}
```
