# Taiji Single-Sphere Architecture (TSA)

## Introduction

Modern large language models are dominated by probability matrices. While the Transformer architecture has achieved remarkable success, its reliance on Softmax (a zero-sum probability game) and high-dimensional feature mixing (QKV) introduces fundamental limitations: numerical instability in long contexts, heavy KV Cache memory footprints, and a structural inability to express "Absolute Silence" (the absence of relevance).

**The Taiji Architecture** is a fundamental reconstruction of the language model's bottom layer. Based on first principles of geometry and physics, we replace probability distributions with a pure physical dynamics engine. 

This repository documents our complete research journey, culminating in **Phase 4 (V20)**, which eliminates both the Softmax mechanism and the final linear output layer.

---

## Research Roadmap

### Phase 1: Single-Sphere Geometric Evolution (Deconstruction & Reconstruction)
*Location: `/phase1_single_sphere`*

*   **Core Idea**: Abolish QKV projections and multi-head mechanisms. Decompose semantic vectors purely into **Direction (Phase/Position)** and **Scalar (Energy/Value)**.
*   **Mechanism**: Attention is calculated via pure geometric phase differences rather than Softmax. The model only absorbs scalar energy (\(R\)) without mixing directions, achieving "Relative Inclusion".

### Phase 2: Mass-Center Integration & The Physics of "Nothing"
*Location: `/phase2_center_of_mass`*

*   **Core Idea**: To resolve the numerical explosion in long texts and physically express "Absolute Silence", we introduce Newtonian mechanics: **Mass-Center Integration**.
*   **Mechanism**: The current token dynamically merges into the global historical center of mass. The state transition is gated by the interference energy.

### Phase 3: Macro-Micro Unified Field (V19)
*Location: `/phase3_unified_field`*

*   **Core Idea**: Center of Mass destroys exact micro-memory (Copy tasks). To fix this, we introduce a multi-order force field to balance global context and local precision.
*   **Mechanism**: Uses a 1st-order macro-force for smooth context and a 3rd-order micro-force for sharp resonance. However, it still relies on a final `nn.Linear` layer for output.

### Phase 4: Pure Physical Emergence (V20) - Current Breakthrough
*Location: `/phase4_physical_emergence`*

*   **Core Idea**: Remove the final linear layer entirely. The next token is physically "emerged" via geometric distance matching on the high-dimensional sphere. True zero-interference is achieved.
*   **Key Results**: No QKV, No Softmax, No Linear Head.

---

## Core Experimental Results (V20)

Tested on a 2-layer, 128-dim model without pre-training:

*   **SST-2 Sentiment Classification**: **72.36%** (Standard Transformer baseline is 71.10%).
*   **Copy Task (Micro-memory)**: Loss drops to **0.0042**.
*   **Zero-Interference State**: If a token is completely orthogonal to history, the state transition step is exactly **0.000000** (Softmax mathematically cannot achieve this).

### Mechanism Comparison

| Dimension | Traditional Transformer | Taiji V20 Architecture |
| :--- | :--- | :--- |
| **Projection Matrices** | Relies on independent \(W_q, W_k, W_v\) | No projection matrices; directly decomposes token vectors |
| **Feature Aggregation** | High-dimensional vector mixing (Feature Pollution) | Pure scalar energy / Force field aggregation |
| **Inference Cache** | Requires caching massive Key and Value matrices | Only requires caching direction and scalar energy |
| **Output Layer** | Final `nn.Linear` + Softmax | Pure geometric distance matching |
| **"Unknown" Expression** | Impossible (Forced probability distribution) | **Absolute Silence (0.000000)** |

---

## Honest Ablation & Limitations

Scientific truth requires acknowledging boundaries. We ran rigorous ablations:

1.  **Macro vs. Micro on SST-2**: Using *only* the 1st-order macro-force achieves **73.51%** on SST-2, slightly outperforming the unified field (72.36%). This proves that sentiment classification requires smooth global context; the sharp 3rd-order micro-force introduces slight local noise to the global smoothing.
2.  **Micro-force on Copy Task**: Without the 3rd-order force, Copy Loss stalls at 0.0102. With it, Loss drops to 0.0042. The micro-force is strictly specialized for exact local memory.
3.  **Information-Theoretic Limit**: In a highly noisy, random Copy task (30% injected random tokens), both macro-only and unified models collapse (Loss ~2.5). This confirms that pure physical force fields cannot predict absolute random chaos.

---

## Code Execution

```bash
# 1. Clone the repository
git clone https://github.com/Wangzhifang199127/Taiji-Single-Sphere.git
cd Taiji-Single-Sphere

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Phase 4 (V20) training and evaluation
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
