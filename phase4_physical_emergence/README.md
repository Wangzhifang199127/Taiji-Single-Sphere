# Taiji Phase 4: Pure Physical Emergence (V20)

## Introduction
In Phase 3 (V19), we unified macro-context and micro-memory using a multi-order force field. However, V19 still relied on a final linear layer (`nn.Linear`) to project the evolved state into the vocabulary space.

Phase 4 (V20) removes this final architectural compromise. **We completely eliminate the linear output layer.** Instead, the final token state evolves into a future trajectory, and the next token is selected purely via geometric distance matching on the high-dimensional sphere. This is true "Physical Emergence."

## Key Innovations

1. **No Final Linear Layer**: The vocabulary embedding matrix acts as the "cosmic star map." The evolved state physically collides with the nearest word.
2. **Adaptive Energy Constraint**: The force applied by the field is scaled by the inverse of the token's energy (`1 / R`). Real words resist external force, while function words are easily carried away.
3. **Absolute Causal Isolation**: Zero-interference (Orthogonal) states remain strictly untouched.

## Experimental Results

### 1. Macro-Understanding (SST-2)
Without pre-training and without a final linear layer:
*   **Taiji V20**: **72.36%**
*   *Standard Transformer Baseline*: *71.10%*

### 2. Micro-Memory (Copy Task)
*   **Copy Loss**: Drops to **0.0042** (near zero).

*(Note: Diagnostics charts are automatically generated in the `assets/` folder by running `main.py`.)*

## Code Execution
```bash
cd phase4_physical_emergence
python main.py