# Module C Structural Motif Discovery Attempts Log

This document records every structural motif discovery (Module C) configuration evaluated in this project, documenting its formulation, motivation, empirical metrics, and downstream outcome.

---

### Attempt 1: Raw Unnormalized Adjacency Degree Pooling
* **Formulation:** Global pooling across windows using raw node degree sums: $\text{pool}_s = \sum_{t} A_t \mathbf{1}$.
* **Hypothesis:** High-degree hub nodes provide a strong temporal signature for motif slot matching.
* **Findings:** 
  * Total edge counts grew monotonically across development ($775 \to 2186$ chemical synapses).
  * Decoder reconstruction loss: Initial MSE $1.34 \to$ Final MSE $0.18$.
  * Slot usage entropy collapsed: Attention was dominated by 1–2 slots tracking raw edge count.
* **Outcome:** **Discarded.** Failed to represent network geometry; captured global developmental density/size rather than recurring structural motifs.

---

### Attempt 2: Unit-$L_2$ Normalized Vector Pooling
* **Formulation:** Pooled vectors were normalized to unit sphere before scaled dot-product attention: $\hat{p}_s = p_s / \|p_s\|_2$.
* **Hypothesis:** Eliminating magnitude prevents scalar scale dominance while preserving angle.
* **Findings:**
  * Attention routing entropy improved ($H(p) \approx 1.84$ vs $\ln(10) \approx 2.30$).
  * Synthetic injected 4-node clique recovery passed ($100\%$ detection).
  * On real connectomes, window embeddings remained in a narrow acute cone (mean pairwise cosine similarity $> 0.98$).
* **Outcome:** **Partial.** Retained as baseline mechanism for global graph pooling in Phase 4.

---

### Attempt 3: Degree-Sum Density Invariance Scaling
* **Formulation:** For adjacency $A \in \mathbb{R}^{N \times N}$, normalized node degree vector $d_{\text{norm}} = \frac{d}{\sum d} \cdot N$.
* **Hypothesis:** Preserves local relative degree heterogeneity (hubs, cliques) while strictly enforcing global density mean of $1.0$.
* **Findings:**
  * Enabled robust synthetic clique detection in alternating windows ($k=3$).
  * Chemical synapses K-sweep across $K \in \{5, 10, 20, 50\}$ yielded reconstruction drop $> 85\%$ without single-slot collapse.
* **Outcome:** **Standardized for Module C standalone pooling.** Integrated into `pcdgm/module_c_motifs.py`.

---

### Attempt 4: Subnetwork / Command Circuit Pooling (AVA / AVB / AVE Analysis)
* **Formulation:** Evaluated motif slot assignment specifically for command interneurons (AVA, AVB, AVE, PVC) vs. full network.
* **Hypothesis:** Command circuit neurons share specialized structural motifs distinct from sensory/motor neurons.
* **Findings:**
  * AVA/AVB/AVE interneurons were assigned to common slots (Slot 6 in Window 0).
  * Degree confound test: Mann-Whitney $U$ test showed early window slot membership was heavily confounded with in-degree ($p = 1.63 \times 10^{-8}$).
* **Outcome:** **Revealed major degree confound.** Proved that unconstrained topological features inadvertently mirror degree hubs.

---

### Attempt 5: Module B GAT Embeddings with Global Graph Pooling
* **Formulation:** Module B 2-layer GATConv with 4 density-normalized topological features (normalized in/out degree, clustering coeff, 2-hop expansion). Window pooled to single graph-level vector $z_s \in \mathbb{R}^{64}$.
* **Hypothesis:** GAT representations capture rich local network geometry beyond degree.
* **Findings:**
  * Module B passed all unit tests (isolated node stability, LayerNorm stability, output shape $[N, 64]$).
  * However, global pooling collapsed the representation into a single shared vector for all 225 neurons.
* **Outcome:** **Identified Broadcast Ambiguity.** Led to the requirement of testing per-neuron motif representations.

---

### Attempt 6: Per-Neuron Motif Vectors (Initial Scaled Dot-Product, $\tau = 1.0$)
* **Formulation:** Each neuron $i$ receives its own motif vector $z_i = a_i D$, where $a_i = \text{softmax}(h_i D^T / \sqrt{d})$.
* **Hypothesis:** Allows neurons to occupy different structural roles and query different motif dictionary slots.
* **Findings:**
  * Across-neuron cosine similarity: **$0.999861$** (mean std: $0.000866$).
  * Across-window cosine similarity: **$0.9999$** (mean drift: $0.001993$).
  * 5-Seed Connectome Benchmark:
    * No Motif: AUROC $0.6536 \pm 0.0067$ | AUPRC $0.6367 \pm 0.0053$
    * Random Untrained: AUROC $0.6523 \pm 0.0052$ | AUPRC $0.6340 \pm 0.0061$
    * Learned PCDGM: AUROC $0.6463 \pm 0.0113$ | AUPRC $0.6264 \pm 0.0121$
* **Outcome:** **Failed to differentiate neurons.** All 225 neurons received effectively identical vectors; performance differences were within random noise.

---

### Attempt 7: Normalized Attention Sharpening without Centering ($\tau = 0.1$)
* **Formulation:** $L_2$-normalized neuron embeddings $\hat{h}_i$ and dictionary vectors $\hat{D}_k$, with $\text{logits}_i = (\hat{h}_i \hat{D}^T) / \tau$, $\tau = 0.1$.
* **Hypothesis:** Sharpening temperature will force neurons into distinct dictionary slots.
* **Findings (Spread Gate on Real Data):**
  * Window 0 to 5 mean cosine similarities: $0.9841 \to 0.9972$.
  * Overall across-neuron cosine similarity: **$0.9920$** ($> 0.95$).
  * Mean attention entropy / $\ln(K)$: **$0.8508$**.
* **Outcome:** **FAILED SPREAD GATE.** Stop condition triggered; training was halted. Embeddings were clustered in a narrow cone so temperature scaling alone did not disperse attention.

---

### Attempt 8: Mean-Centered Embeddings with Temperature $\tau = 0.1$ (Attempt A)
* **Formulation:** Subtract batch/window mean embedding over all 225 neurons: $\tilde{h}_i = h_i - \frac{1}{N}\sum_j h_j$. Then $L_2$-normalize $\hat{h}_i = \tilde{h}_i / \|\tilde{h}_i\|_2$ and $\hat{D}_k = D_k / \|D_k\|_2$. Compute attention with fixed $\tau = 0.1$: $a_i = \text{softmax}(\hat{h}_i \hat{D}^T / \tau)$, $z_i = a_i \hat{D}$.
* **Hypothesis:** Centering removes the common mean vector, dispersing neuron directions across the unit sphere.
* **Findings:**
  * **Spread Gate (Real Connectome Data):**
    * Mean across-neuron cosine similarity: **$0.4049$** ($\le 0.95$ gate passed).
    * Mean feature std across neurons: **$0.0464$**.
    * Mean attention entropy / $\ln(K)$: **$0.7507$**.
    * Degree check: Mean Pearson $r = 0.0602$, $R^2 = 0.0270$ (motif representation is non-confounded with degree).
  * **5-Seed Paired Connectome Benchmark:**
    * 1. No Motif Module: AUROC **$0.6547 \pm 0.0072$** | AUPRC **$0.6378 \pm 0.0061$** | Val AUROC **$0.6736 \pm 0.0066$**
    * 2. Random Untrained ($D$ frozen): AUROC **$0.6441 \pm 0.0183$** | AUPRC **$0.6278 \pm 0.0196$** | Val AUROC **$0.6700 \pm 0.0134$**
    * 3. Learned Patterns (PCDGM): AUROC **$0.6545 \pm 0.0065$** | AUPRC **$0.6362 \pm 0.0067$** | Val AUROC **$0.6781 \pm 0.0034$**
  * **Statistical Significance (Paired $t$-tests):**
    * Learned vs. No-Motif: Mean paired diff = $-0.0003$, $t = -0.0413$, $p = 0.9690$ (not significant).
    * Learned vs. Random: Mean paired diff = $+0.0104$, $t = 1.4853$, $p = 0.2117$ (not significant).
* **Outcome:** **Successfully dispersed neuron motifs ($0.4049$ cosine similarity) without degree confounding ($R^2 = 0.027$), but the learned motif dictionary does not statistically outperform plain TGN on link prediction (null result).**

---

### Benchmark Script Rewrite, Determinism Hardening & Rerun
* **Code Rewrite & Deterministic Rerun:** `run_phase5_training.py` was rewritten to pair all 3 variants (`no_motif`, `random_untrained`, `learned`) across the exact same seeded mini-batches and model initializations. Deterministic execution was strictly enforced (`torch.use_deterministic_algorithms(True)`, `random.seed(s)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`), eliminating slight batch-sampling float drift and guaranteeing bit-identical test outputs across repeated runs.
* **Status of Module B/C/D/E Architecture & "When" Prediction:**
  * **Plain Statement:** The full modular `PCDGM` architecture combining Modules B, C, D, and E (`pcdgm/model.py`) and the Module E "when" (temporal interval / timing) prediction head have **not been evaluated on real connectome data**.
  * The Phase 5 connectome benchmark evaluated link prediction ("where") on real Witvliet connectome data using the baseline `ConnectomePCDGM` (TGN-based architecture with motif slot attention). The full end-to-end `PCDGM` model (incorporating GAT Module B, temporal trajectory interpolation, fused predictor Module E, and multi-task loss with $L_{\text{timing}}$) has only been validated in synthetic unit/integration test suites and has not been run or benchmarked on real connectome data.
