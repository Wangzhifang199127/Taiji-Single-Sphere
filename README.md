# Taiji Architecture: From QKV Deconstruction to a Pure Physical State Transition Engine

## Introduction

Modern large language models are dominated by probability matrices and statistical fitting. While the Transformer architecture has achieved remarkable success, its reliance on Softmax (a zero-sum probability game) and high-dimensional feature mixing (QKV) introduces fundamental limitations: inevitable numerical explosion in long contexts, heavy KV Cache memory footprints, and a structural inability to express "Absolute Silence" (the absence of relevance), which is a primary source of hallucination.

**The Taiji Architecture** is a fundamental reconstruction of the language model's bottom layer. Based on the first principles of geometry and physics—embodying the unity of *Image, Number, and Principle* (象数理一体)—we replace probability distributions with a pure physical dynamics engine.

This repository documents the complete research journey, structured into two evolutionary phases.

---

## Research Roadmap

### Phase 1: Single-Sphere Geometric Evolution (Deconstruction & Reconstruction)
*Location: `/phase1_single_sphere`*

*   **Core Idea**: Abolish QKV projections and multi-head mechanisms. Decompose semantic vectors purely into **Direction (Phase/Position)** and **Scalar (Energy/Value)**.
*   **Mechanism**: Attention is calculated via pure geometric phase differences (Cosine Similarity) rather than Softmax. The model only absorbs scalar energy (\(R\)) without mixing directions, achieving "Relative Inclusion".
*   **Key Achievement**: Successfully validated the geometric evolution of relations and energy, demonstrating the emergence of "Energy Waves" (long-tail distributions). Theoretical derivation shows a **~50% reduction** in KV Cache memory by eliminating the high-dimensional \(V\) matrix.

### Phase 2: Mass-Center Integration & The Physics of "Nothing" (Macro-Dynamics)
*Location: `/phase2_center_of_mass`*

*   **Core Idea**: To resolve the numerical explosion in long texts and physically express "Absolute Silence", Phase 2 introduces Newtonian mechanics: **Mass-Center Integration**.
*   **Mechanism**: The current token dynamically merges into the global historical center of mass. The state transition is gated by the interference energy.
*   **Key Achievement**: 
    *   **Macro-Performance**: Outperforms the standard Transformer in SST-2 sentiment classification without pre-training (**73.62% vs 71.10%**), acting as a natural physical regularizer.
    *   **The "Orthogonal Island" Test**: By ensuring strict Gram-Schmidt orthogonality, an irrelevant token achieves an absolute state transition step size of **0.000000**. This proves the architecture's unique physical ability to express "Nothing" (Zero-Interference), a feat mathematically impossible for standard Softmax.

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
 ┗ 📜 README.md                   # This document
```

---

## Key Discoveries & Validations

| Feature | Traditional Transformer | Taiji Architecture |
| :--- | :--- | :--- |
| **Attention Mechanism** | Softmax (Probability / Zero-sum) | Absolute Geometric Field (Phase Difference) |
| **Information Flow** | High-dimensional feature mixing (Pollution) | Pure scalar energy transfer (Relative Inclusion) |
| **"Unknown" Expression** | Impossible (Forced distribution causes hallucination) | **Absolute Silence (0.000000 step size)** |
| **KV Cache** | Huge (Caches \(K\) and \(V\)) | Reduced by ~50% (Only caches direction & \(R\)) |
| **Macro Understanding** | Overfits; relies on deep layers | Natural regularizer; captures macro-potential via Center of Mass |

---

## Future Work

The current repository represents a Proof of Concept at a small parameter scale. Our next research frontier focuses on a **Universal Capture Mechanism** (e.g., Wave-Particle Duality) to unify macro-context and micro-generation, completely bypassing the limitations of Softmax.

---

## Citation

```bibtex
@misc{taiji_architecture_2024,
  author = {Wang Zhifang},
  title = {Taiji Architecture: From QKV Deconstruction to a Pure Physical State Transition Engine},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Wangzhifang199127/Taiji-Single-Sphere}}
}
```