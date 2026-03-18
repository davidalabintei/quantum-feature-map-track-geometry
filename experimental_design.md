# Quantum Feature Map for Track Geometry Prediction: Experimental Design

## 1. Objective

Investigate whether a deterministic quantum feature map applied to track geometry inspection data produces enhanced features that improve classical ML prediction of future geometry values, compared to classical-only approaches.

**Primary research question:** Does encoding track geometry parameters into a quantum feature space — capturing nonlinear and periodic cross-parameter interactions via entanglement — yield better one-month-ahead predictions than classical models operating on raw or classically engineered features?

**Secondary research questions:**
- Which quantum features (individual qubit terms vs. pairwise vs. three-body correlators) carry the most predictive information?
- Which geometry parameter interactions does the quantum circuit encode that classical feature engineering misses?
- How does circuit depth affect prediction quality?

---

## 2. Data

**Source:** Monthly track geometry inspection data, approximately 24–48 months per segment.

**Parameters (features):**
- Profile (vertical alignment)
- Alignment (horizontal alignment)
- Gauge
- Warp
- Twist
- Crosslevel

**Target variable:** Next month's geometry values (one-step-ahead forecasting). Each parameter can be predicted independently or jointly.

**Temporal structure:** Each track segment has a time series of 24–48 monthly observations across 6 parameters.

### 2.1 Data Preparation

**Windowed input construction:**
For each segment at time t, construct the input feature vector from:
- Current values: profile(t), alignment(t), gauge(t), warp(t), twist(t), crosslevel(t) — 6 features
- First differences (rate of change): Δprofile(t) = profile(t) − profile(t−1), etc. — 6 features
- Optionally, second differences or rolling statistics (e.g., 3-month rolling std) — 6 features

This gives a base feature vector of 12–18 classical features per observation.

**Target:** geometry values at t+1 (or the change from t to t+1, depending on what proves more learnable).

**Train/test split:** Time-based split. Use the first ~80% of the time series for training and the last ~20% for testing. Do NOT shuffle — respect temporal ordering to avoid data leakage.

**Scaling for quantum encoding:** All features must be scaled to [0, π] for angle encoding. Use min-max scaling computed on the training set only, applied to both train and test.

---

## 3. Quantum Circuit Design

### 3.1 Feature Map Architecture

**Primary circuit: ZZ Feature Map**

For n input features mapped to n qubits:

1. **Layer 1 — Hadamard:** Apply H gate to all qubits (creates superposition)
2. **Layer 2 — Encoding:** Apply Rz(xᵢ) rotation to qubit i, where xᵢ is the scaled feature value
3. **Layer 3 — Entanglement:** For selected qubit pairs (i, j), apply CNOT followed by Rz(xᵢ · xⱼ), then CNOT again. This encodes pairwise feature interactions.
4. **Repeat:** Steps 1–3 can be repeated (circuit depth d) for richer nonlinear functions.

**Entanglement topologies to test:**
- **Linear/nearest-neighbour:** Pairs (1,2), (2,3), (3,4), (4,5), (5,6). Fewest entangling gates, fastest to simulate. Qubit ordering matters — experiment with different orderings of geometry parameters.
- **Full pairwise:** All 15 pairs for 6 qubits. Maximum expressiveness but more expensive. Captures all cross-parameter interactions.
- **Physically motivated:** Only entangle parameters with known physical coupling — e.g., (profile, twist), (gauge, crosslevel), (alignment, gauge), (warp, twist). This embeds domain knowledge into the circuit structure.

**Circuit depths to test:** d = 1, 2, 3. Deeper circuits produce higher-frequency Fourier components.

### 3.2 Qubit-to-Parameter Assignment (for nearest-neighbour topology)

Since nearest-neighbour entanglement only couples adjacent qubits, the ordering of geometry parameters on qubits determines which interactions are captured. Test at least two orderings:

- **Ordering A (physical coupling):** Profile – Twist – Warp – Crosslevel – Gauge – Alignment
  (groups parameters that are physically related as neighbours)
- **Ordering B (measurement grouping):** Profile – Alignment – Gauge – Crosslevel – Twist – Warp

Compare results to assess sensitivity to qubit ordering.

### 3.3 Measurements

Extract three tiers of features from the quantum circuit:

**Tier 1 — Individual expectations (6 features):**
⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩, ⟨Z₄⟩, ⟨Z₅⟩, ⟨Z₆⟩

Each is a nonlinear (trigonometric) function of all input features, shaped by entanglement.

**Tier 2 — Two-body correlators (up to 15 features):**
⟨ZᵢZⱼ⟩ for selected or all pairs.

These capture pairwise parameter interactions. Crucially, ⟨ZᵢZⱼ⟩ ≠ ⟨Zᵢ⟩·⟨Zⱼ⟩ — the difference encodes quantum correlations.

**Tier 3 — Three-body correlators (up to 20 features):**
⟨ZᵢZⱼZₖ⟩ for selected or all triplets.

These capture three-way parameter interactions that are expensive to engineer classically.

**Measurement configurations to test:**
- Tier 1 only (6 features) — minimal quantum features
- Tier 1 + Tier 2 (21 features) — recommended starting point
- Tier 1 + Tier 2 + Tier 3 (41 features) — full feature extraction

---

## 4. Classical ML Pipeline

### 4.1 Models

Apply each of the following to every feature configuration:
- **Ridge Regression** — linear baseline
- **XGBoost** — gradient boosting (strong nonlinear baseline)
- **CatBoost** — gradient boosting (handles categorical-like discretised features well)

Hyperparameters should be tuned via time-series cross-validation (expanding window) on the training set.

### 4.2 Feature Configurations (Experimental Conditions)

| Label | Description |
|-------|-------------|
| **C1 — Raw** | Classical ML on raw geometry features + temporal derivatives (12–18 features) |
| **C2 — Poly** | Classical ML on raw features + polynomial interaction terms (xᵢ·xⱼ for all pairs) |
| **C3 — Trig** | Classical ML on raw features + classical trigonometric features: cos(xᵢ), cos(xᵢ)·cos(xⱼ), sin(xᵢ)·sin(xⱼ) |
| **Q1 — ZZ-linear-d1** | Quantum features from ZZ map, nearest-neighbour entanglement, depth 1, Tier 1+2 measurements |
| **Q2 — ZZ-full-d1** | Quantum features from ZZ map, full pairwise entanglement, depth 1, Tier 1+2 measurements |
| **Q3 — ZZ-full-d2** | Quantum features from ZZ map, full pairwise entanglement, depth 2, Tier 1+2 measurements |
| **Q4 — ZZ-full-d1-T3** | Same as Q2 but with Tier 1+2+3 measurements (41 features) |
| **Q5 — ZZ-physical-d1** | Quantum features from ZZ map, physically motivated entanglement, depth 1, Tier 1+2 measurements |
| **H1 — Hybrid** | Quantum features (best Q configuration) concatenated with classical temporal features |

**C3 is critical** — it tests whether the improvement (if any) comes from the periodic function structure specifically, or from the quantum entanglement encoding. If Q configurations beat C1 and C2 but not C3, the value is in the trigonometric basis, not the quantum circuit. If Q beats C3, the quantum circuit is producing something the classical trigonometric features don't capture.

### 4.3 Per-Parameter vs. Joint Prediction

- **Per-parameter:** Train a separate model for each geometry parameter (e.g., predict profile(t+1) from all features at t). Six separate models.
- **Joint prediction:** Use multi-output regression to predict all six parameters simultaneously. This tests whether the quantum features help capture cross-parameter predictive relationships.

Start with per-parameter prediction for simplicity, then test joint prediction.

---

## 5. Evaluation

### 5.1 Metrics

- **RMSE** (root mean squared error) — primary metric, directly comparable to Tamturk et al.
- **MAE** (mean absolute error) — less sensitive to outliers
- **R²** (coefficient of determination) — interpretability of explained variance

Report all three for every model × feature configuration combination.

### 5.2 Statistical Significance

Run each experiment across multiple track segments. Use paired statistical tests (e.g., Wilcoxon signed-rank test) to determine whether performance differences between classical and quantum configurations are statistically significant, not just numerical.

### 5.3 Prediction Horizons

Primary: t+1 (one month ahead).
Secondary: t+3 and t+6 (three and six months ahead, using recursive or direct multi-step forecasting). This tests whether quantum features provide more durable predictive information over longer horizons.

---

## 6. Interpretability Analysis

If quantum feature configurations outperform classical ones, conduct the following analyses to understand why.

### 6.1 Feature Importance (SHAP)

Compute SHAP values for the best-performing quantum-enhanced model. Since each quantum feature maps to a specific qubit (parameter) or qubit pair (parameter interaction), SHAP values directly indicate which geometry parameters and parameter interactions drive predictions.

**Deliverable:** Ranked list of quantum features by importance, mapped back to physical parameter names and interactions.

### 6.2 Circuit Ablation

Systematically remove entangling gates between specific qubit pairs and re-run prediction. Measure the change in RMSE.

**Deliverable:** A matrix showing performance degradation when each entangling gate is removed. This reveals which parameter couplings the circuit is usefully encoding.

### 6.3 Fourier Spectrum Analysis

Decompose the quantum feature map into its Fourier series representation (following the framework of Schuld, Sweke & Meyer, 2021). Compare the accessible frequency spectrum of the circuit against the dominant frequency content in the geometry data (from spectral analysis of the time series).

**Deliverable:** Side-by-side comparison of circuit Fourier spectrum and data Fourier spectrum. Alignment between the two provides a principled explanation for why the quantum map suits this data.

### 6.4 Quantum vs. Classical Feature Comparison

For the most important quantum features identified by SHAP, construct the closest classical analog (e.g., if ⟨Z₂Z₅⟩ is important, compute cos(x₂)·cos(x₅) classically). Test whether replacing quantum features with their classical analogs recovers the same performance.

**Deliverable:** Feature-by-feature comparison quantifying the "quantum surplus" — the performance gap between quantum features and their closest classical equivalents.

---

## 7. Implementation

### 7.1 Tools

- **Qiskit** (IBM) — quantum circuit simulation, ZZ feature map implementation
- **Qiskit Aer simulator** — noise-free statevector simulation for deterministic results
- **scikit-learn** — Ridge Regression, preprocessing, evaluation metrics
- **XGBoost / CatBoost** — gradient boosting implementations
- **SHAP** — feature importance analysis
- **Python (NumPy, pandas, matplotlib)** — data handling and visualisation

### 7.2 Simulation Notes

Since the feature map is deterministic and we use expectation values (not shot-based sampling), the statevector simulator gives exact results with no randomness — unlike the QRC approach in Tamturk et al. This eliminates the need for multiple runs and averaging, making results fully reproducible.

---

## 8. Expected Outcomes and Contribution

**If quantum features improve prediction:** The contribution is demonstrating that quantum feature maps capture physically meaningful nonlinear interactions between track geometry parameters that improve degradation forecasting, with a clear interpretability framework mapping circuit structure to parameter couplings.

**If quantum features match but don't exceed classical trigonometric features (C3):** The contribution is showing that the periodic function structure is what matters, not the quantum implementation — which is still a useful finding for the track geometry community, pointing toward trigonometric feature engineering as an underexplored approach.

**If quantum features don't help:** The contribution is a rigorous negative result with analysis of why the geometry data structure doesn't align with what quantum feature maps produce — valuable for the quantum ML community in understanding domain applicability.
