# THE BOOK
## Predicting Synaptic Connectivity Change in Developmental Connectomes
### A Complete, Self-Contained Specification for Research and AI-Assisted Implementation

---

# HOW TO USE THIS BOOK

This book has seven parts. Part I is the research itself (why, what, hypotheses). Parts II–VI are for building it. Part VII is how to actually work with AI on this without the whole thing quietly going wrong.

**The single most important rule in this entire book:** never hand an AI coding assistant this whole document and ask for the whole project in one response. Work through Parts III–V module by module, testing between each. Everything in Part VII explains why and exactly how.

---

# PART I — THE RESEARCH

## 1. Title

**Predicting the Timing and Locus of Synaptic Connectivity Change in Developmental Connectomes via Structural Motif Discovery**

## 2. Abstract

Understanding how a nervous system wires itself during development remains one of the open problems in connectomics — we know the starting and ending states reasonably well, but the *process* of getting there is much harder to pin down, especially when the only data we have are a handful of snapshots spread across development. In this work, we ask a simple question: given just eight reconstructed connectomes of *C. elegans* spanning birth to adulthood, can we say something meaningful about *when*, not just *whether*, individual synaptic connections form, strengthen, or disappear?

We approach this by reconstructing a continuous developmental trajectory from these sparse snapshots — using both the real known collection timestamps and, separately, a blind structural-inference method validated against those real timestamps — and then modeling synaptic change as a sequence of recurring structural motifs unfolding along that trajectory, rather than as independent, unordered snapshot-to-snapshot jumps. A learned motif dictionary captures these recurring patterns, and a stochastic event model, adapted from recent work on motif-driven temporal graph prediction, uses them to predict both which connections change and roughly when. We compare this against standard temporal graph baselines and ablations that strip out the motif structure entirely, to test whether the patterns are doing real predictive work or simply adding parameters.

We are explicit throughout about what is and is not new here: the underlying techniques are adapted from other domains; the contribution is applying and validating them on structural connectome data, where no predictive model of this kind currently exists.

## 3. Motivation and Problem Statement

Developmental connectomics can describe *what* a nervous system's wiring looks like at a few sparse points in time, but not the finer-grained process of *how* it gets there. The most complete resource for studying this — Witvliet et al.'s (2021) eight-timepoint reconstruction of the *C. elegans* connectome — has so far been used predictively only for *cell lineage* events, by DevoTG (Gayen, 2026). The connectome itself, in that work, is treated descriptively: connections are classified post hoc as stable, developmental, or variable, but no model predicts *future* connectivity change. DevoTG's authors state directly that training a model to predict synapse formation on the connectome data itself is a natural next step. This book takes up that step.

**Primary research question:** Can recurring structural motifs, discovered from a connectome's own developmental history, serve as a compact and genuinely predictive signal for which synaptic connections change — and approximately when — across development?

**Explicitly out of scope:** cell lineage/division prediction (already solved by DevoTG); any use of neural activity data (calcium imaging, electrophysiology); any organism other than *C. elegans*; any claim that the underlying motif-timing mechanism itself is a new invention.

## 4. Related Work and Honest Novelty Statement

- **Temporal Graph Networks** (Rossi et al., 2020): standard toolkit for dynamic graph link prediction. Used as baselines, not as the contribution.
- **DevoTG** (Gayen, 2026): the direct predecessor. Same dataset, predicts lineage events, analyzes but does not predict connectome-level change, explicitly names connectome-level prediction as unaddressed future work.
- **STEP** (arXiv 2603.05874, 2026): motif-transition-based stochastic event prediction in continuous time for temporal graphs — the closest prior art for the which+when prediction mechanism. Adapted here, not invented.
- **Temporal motif representation learning** (2026) and **TempME**: establish that learned/counted temporal motifs are informative features for dynamic graph link prediction generally. Required as a baseline.
- **Pseudotime-informed temporal graph learning** (arXiv 2605.28659, 2026, applied to gene regulatory networks): establishes reconstructing a continuous trajectory from sparse biological snapshots for downstream graph learning — closest prior art for the trajectory-reconstruction component.

| Component | Status |
|---|---|
| Motif-based conditioning for graph event timing | NOT novel — cite STEP directly |
| Trajectory reconstruction from sparse graph snapshots | NOT novel — cite the GRN pseudotime paper |
| Application to structural synaptic connectivity prediction on a developmental connectome | **Novel** — this is the actual claim |
| Graph-level (whole-graph-to-whole-graph) structural trajectory inference validated against real timestamps | Unverified — treat as tentative, re-check at write-up time |
| Whether H1 (below) is true | Unknown until experiments run — empirical, not assumed |

**The honest claim:** application-level and combinatorial novelty — combining two established technique classes, neither previously applied to this domain or this prediction target — not the invention of a new underlying mechanism.

## 5. Formal Hypotheses

**H1 (primary):** `P(E_{t+τ} changes | z_t, G_t) > P(E_{t+τ} changes | G_t)` — conditioning on the learned motif representation `z_t` measurably improves prediction over structure alone, at a statistically significant margin across ≥5 seeds.

**H2 (secondary):** Blind, structure-only trajectory inference over the 8 snapshots recovers the true developmental ordering and approximate relative timing at above-chance accuracy, checked against real known hours.

**H3 (exploratory):** Motifs discovered as preceding connectivity change near AVA/AVB/AVE are qualitatively consistent with independently reported reinforcement patterns for these hub neurons.

**Rule:** every experiment maps to exactly one of H1/H2/H3. A negative result on any hypothesis, rigorously demonstrated, is a valid finding — not a project failure.

## 6. Data

Witvliet et al. (2021), *C. elegans* developmental connectome:
- 4× L1 larvae: 0h, 5h, 8h, 16h post-hatching
- 1× L2: 23h
- 1× L3: 27h
- 2× young adult replicates: 45h, 45h
- ~225 neurons, 858–2,496 connections per timepoint

**Access:**
- `git clone https://github.com/Jayadratha/DevoTG_GSoC` (research branch) — pre-parsed, also needed for Phase 1 regardless.
- `pip install cect` (OpenWorm Connectome Toolbox) → `from cect.WitvlietDataReader import WitvlietDataReader1` through `8` — independent cross-check.
- Zenodo DOI `10.5281/zenodo.5637219` — original raw archive, for verifying counts against the paper directly.

No neural activity data is used anywhere in this project.

## 7. Expected Contributions

1. First predictive (not merely descriptive) model of structural synaptic connectivity change on a developmental connectome.
2. A trajectory-reconstruction method from n=8 real snapshots, validated against real timestamps — a check rarely possible in the broader pseudotime literature.
3. An empirical answer, either direction, to whether structural motifs carry genuine predictive signal (H1).
4. A domain-specific interpretability case study (H3), clearly labeled exploratory.
5. Independently: the first structure-only future-connectivity baseline on this dataset (Phase 2), regardless of whether the full model beats it.

## 8. Limitations (keep in the paper, do not soften at write-up time)

- Single organism, single dataset, n=8 timepoints — generalization unverified.
- Timing predictions bounded by real snapshot resolution; no continuous-hour-level precision claim is valid.
- H3 is correlational and exploratory; no causal biological claim is made.
- Underlying mechanisms (motif-conditioned continuous-time events, trajectory reconstruction) are adapted, not invented; novelty is scoped to application and combination.

---

# PART II — OBJECTIVE & HARD RULES (paste alongside any AI prompt)

## Objective (single source of truth)

Predict which synaptic connections change (form, strengthen, weaken, disappear) across *C. elegans* development, and estimate roughly when, using recurring structural motifs discovered from the connectome's own developmental history — evaluated on the 8 real Witvliet snapshots, no activity data.

## Hard Rules

1. Temporal splits only. Never shuffle across time.
2. The two 45h replicates are not sequential — same developmental stage, two animals.
3. No activity data, anywhere, ever, in this version of the project.
4. 5 seeds minimum for any reported ablation number, propagated to data shuffling, model init, AND dictionary init separately.
5. Report `L_edge` separately from total loss, always.
6. `a_t`/`a_s` (motif attention weights) must be a retrievable output, not internal-only.
7. Any surprisingly strong result gets a leakage check before being trusted.
8. DevoTG's reproduction number (Phase 1) must match before any novel module is trusted.

---

# PART III — ARCHITECTURE

## Pipeline

```
8 real connectome snapshots E_1..E_8 (known hours: 0,5,8,16,23,27,45,45)
        │
        ▼
MODULE A: Trajectory Reconstruction  → continuous developmental axis s ∈ [0,45]
        │
        ▼
MODULE B: Structural Encoder         → H_s  Tensor[N, 64]
        │
        ▼
MODULE C: Motif Discovery            → z_s, a_s, D (dictionary)
        │
        ▼
MODULE D: Graph Fusion               → H_s'  Tensor[N, 100]
        │
        ▼
MODULE E: Which+When Predictor       → p_ij, t_ij
```

Two parallel codebases: `baseline/` (plain TGN, Phase 1–2) must pass its checks before `pcdgm/` (Modules A–E) is built.

## Module A — Trajectory Reconstruction

**A1 (primary): Direct interpolation.** Real hours are known — interpolate edge-existence probability between bracketing real snapshots using monotonic interpolation (e.g. PCHIP, not linear, to avoid overshoot). Fixed, deterministic in v0. Average the two 45h replicates for the interpolation target; keep both as separate eval samples at test time.

**A2 (secondary, novelty-relevant): Blind structural trajectory inference.**
1. Compute pairwise graph-to-graph structural distance between all 8 snapshots (Weisfeiler-Lehman kernel or GNN-embedding L2 distance), blind to timestamps.
2. Build an 8-node meta-graph from these distances.
3. Infer trajectory/ordering via diffusion/random-walk over the meta-graph.
4. Validate (mandatory — this is H2): reveal real hours, compute Spearman correlation between inferred and real order, and between inferred and real relative spacing.

Report both A1 and A2 — A1 is the working pipeline; A2's validation is a standalone, citable finding regardless of what's used downstream.

## Module B — Structural Encoder

Input: node degree, local clustering coefficient, k-hop neighborhood structure (no activity features — none exist). 2-layer GATConv, 2 heads, hidden 100 (matches DevoTG for comparability).

```python
H_s = GATConv2(structural_features(E_hat_s), edge_index=E_hat_s)  # → [N, 64]
```

## Module C — Structural Motif Discovery (core novel-application component)

Window width `k=3` (NOT 10 — only 8 total snapshots exist; this default matters).

```
pool_s      = mean_pool(H_{s-k:s})                         # [64]
a_s         = softmax( (pool_s · D^T) / sqrt(64) )          # [K], K=50 to start, sweep {5,10,20,50}
z_s         = a_s · D                                        # [64]

L_pattern    = MSE( decoder(z_s), pool_s )                  # reconstruction, full window, not last-step-only
L_diversity  = -sum_k( H( mean_s(a_s)_k ) )                  # entropy bonus, prevents collapse
```

`D ∈ R^{K×64}`, learnable, random init in the primary run (data-derived init only as an explicit ablation variant).

**Given only 8 snapshots, dictionary collapse risk is much higher than in data-rich settings — K sweep is required, not optional.**

## Module D — Graph Fusion

```
H_s' = GATConv2( concat(H_s, broadcast(z_s)), edge_index=E_hat_s )   # → [N, 100]
```

Assert `z_s.shape[0] == N` after every broadcast — catches the "one pattern per graph vs. one per node" confusion before it silently corrupts everything downstream.

## Module E — Which + When Predictor

```
p_ij = sigmoid( W3 · ReLU( W1·h_i' + W2·h_j' + W4·z_s ) )     # which

t_ij = s + Δ_ij,   Δ_ij ~ learned stochastic delay distribution conditioned on (h_i', h_j', z_s)   # when
```

**Resolution honesty rule:** ground truth is 8 discrete points, not continuous — report "correct snapshot interval: yes/no" as the primary timing metric. Any finer-grained hour-level MAE is illustrative only and must be explicitly caveated.

## Full Loss

```
L = L_edge + λ·L_pattern + β·L_diversity + γ·L_timing
```
Log all four terms separately, every step.

## Hyperparameter Defaults

| Param | Default | Note |
|---|---|---|
| Encoder/fusion hidden dim | 64 / 100 | matches DevoTG scale |
| K (dictionary slots) | sweep {5,10,20,50} | do not fix at 50 given n=8 |
| Window k | 3 | data-scale appropriate, not 10 |
| λ, β, γ | 1.0, 0.1, 1.0 | starting point; retune from logged per-term losses |
| Seeds | 5 minimum | all three init points varied |
| GAT heads | 2 | matches DevoTG |

---

# PART IV — ROADMAP

**Phase 0** (days): clone DevoTG repo, load and inspect all 8 snapshots, confirm 225 neurons / 858–2,496 edges matches the paper.

**Phase 1:** reproduce DevoTG's lineage baseline exactly — target AUC 0.839 ± 0.007. Pipeline-correctness gate; do not proceed until it matches or the mismatch is understood.

**Phase 2** (new — DevoTG never did this): plain TGN/static-GNN trained directly on the 8 connectome snapshots predicting `E_{t+1}` from `E_{≤t}`. Legitimate, reportable result even standalone.

**Phase 3:** Module A, both A1 and A2, H2 validated and reported as its own table.

**Phase 4:** Module C, synthetic sanity-checked (inject a known repeating motif, confirm recovery) before touching real data.

**Phase 5:** Module C + D + E assembled.

**Phase 6:** full baseline ladder (persistence → static GNN → plain TGN → motif-representation baseline → proposed model) and full ablation set (with/without pattern module, learned vs. random motifs, K sweep, window sweep, A1 vs. A2), H1 tested.

**Phase 7:** interpretability case study against AVA/AVB/AVE, H3 tested.

**Phase 8:** write-up.

---

# PART V — REPOSITORY STRUCTURE, FIXTURES, AND TESTS

## Structure

```
project/
├── external/devotg/          # cloned, untouched, Phase 1 reference only
├── data/
│   ├── loader.py
│   └── test_loader.py
├── baseline/
│   ├── tgn.py
│   └── test_tgn.py
├── pcdgm/
│   ├── module_a_trajectory.py
│   ├── module_b_encoder.py
│   ├── module_c_motifs.py
│   ├── module_d_fusion.py
│   ├── module_e_predictor.py
│   └── test_*.py
├── training/
│   ├── loop.py
│   └── test_loop.py
├── ablations/
│   └── run_ablations.py
└── conftest.py
```

## Shared Fixtures (`conftest.py`)

```python
import pytest
import torch

N_NEURONS = 20
K_TIMEPOINTS = 8
REAL_HOURS = [0, 5, 8, 16, 23, 27, 45, 45]

@pytest.fixture
def synthetic_snapshots():
    torch.manual_seed(0)
    snapshots = []
    base_prob = torch.rand(N_NEURONS, N_NEURONS) * 0.1
    for t in range(K_TIMEPOINTS):
        drift = 0.02 * t
        prob = torch.clamp(base_prob + drift * torch.rand(N_NEURONS, N_NEURONS), 0, 1)
        adj = (torch.rand(N_NEURONS, N_NEURONS) < prob).float()
        adj.fill_diagonal_(0)
        snapshots.append(adj)
    return snapshots

@pytest.fixture
def real_hours():
    return REAL_HOURS

@pytest.fixture
def injected_motif_snapshots():
    """A fixed 4-node clique appears at neurons [0,1,2,3] every OTHER snapshot —
    used to test whether Module C recovers a known motif."""
    torch.manual_seed(1)
    snapshots = []
    for t in range(K_TIMEPOINTS):
        adj = (torch.rand(N_NEURONS, N_NEURONS) < 0.05).float()
        adj.fill_diagonal_(0)
        if t % 2 == 0:
            adj[0:4, 0:4] = 1.0
            adj[0:4, 0:4].fill_diagonal_(0)
        snapshots.append(adj)
    return snapshots
```

## Module 0 Tests (`data/test_loader.py`)

```python
import torch
from data.loader import temporal_split, load_snapshot, validate_node_consistency

def test_no_temporal_leakage():
    """The single most important test in this project."""
    train_idx, val_idx, test_idx = temporal_split(n_timepoints=8, real_hours=[0,5,8,16,23,27,45,45])
    assert max(train_idx) < min(val_idx)
    assert max(val_idx) < min(test_idx)

def test_replicate_snapshots_not_sequential():
    train_idx, val_idx, test_idx = temporal_split(n_timepoints=8, real_hours=[0,5,8,16,23,27,45,45])
    idx_45h = [i for i, h in enumerate([0,5,8,16,23,27,45,45]) if h == 45]
    splits = {i: ('train' if i in train_idx else 'val' if i in val_idx else 'test') for i in idx_45h}
    assert len(set(splits.values())) == 1

def test_node_id_consistency(synthetic_snapshots):
    node_ids_per_snapshot = [list(range(20)) for _ in range(8)]
    assert validate_node_consistency(node_ids_per_snapshot)

def test_neuron_and_edge_counts_match_paper():
    import pytest
    pytest.skip("enable once real data is loaded — do not delete this test")
```

## Module A Tests (`pcdgm/test_module_a_trajectory.py`)

```python
from scipy.stats import spearmanr
from pcdgm.module_a_trajectory import interpolate_A1, infer_trajectory_A2

def test_a1_interpolation_shape(synthetic_snapshots, real_hours):
    interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
    mid_point = interp_fn(s=2.5)
    assert mid_point.shape == synthetic_snapshots[0].shape

def test_a2_is_blind_to_timestamps(synthetic_snapshots):
    order = infer_trajectory_A2(synthetic_snapshots)  # NOTE: no real_hours argument
    assert len(order) == len(synthetic_snapshots)

def test_a2_recovers_real_order_above_chance(synthetic_snapshots, real_hours):
    """H2. A failure here is a valid, reportable result — do not weaken the
    threshold just to force a pass."""
    inferred_order = infer_trajectory_A2(synthetic_snapshots)
    rho, pval = spearmanr(inferred_order, real_hours)
    assert rho > 0.3, f"H2 fails to clear even a weak bar: rho={rho}"
```

## Module B Tests (`pcdgm/test_module_b_encoder.py`)

```python
import torch
from pcdgm.module_b_encoder import StructuralEncoder

def test_output_shape(synthetic_snapshots):
    enc = StructuralEncoder(hidden_dim=64)
    H = enc(synthetic_snapshots[0])
    assert H.shape == (20, 64)

def test_no_batchnorm_used():
    enc = StructuralEncoder(hidden_dim=64)
    for m in enc.modules():
        assert not isinstance(m, torch.nn.BatchNorm1d)

def test_isolated_node_no_nan():
    isolated_graph = torch.zeros(20, 20)
    enc = StructuralEncoder(hidden_dim=64)
    H = enc(isolated_graph)
    assert not torch.isnan(H).any()
```

## Module C Tests (`pcdgm/test_module_c_motifs.py`) — highest scrutiny

```python
import torch
import pytest
from pcdgm.module_c_motifs import MotifDictionary

def test_recovers_injected_motif(injected_motif_snapshots):
    """If this fails, do not proceed to real data — the mechanism is broken."""
    pytest.skip("requires a short training loop — implement as integration test")

def test_dictionary_collapse_detection():
    model = MotifDictionary(K=10, dim=64, window=3)
    fake_attention = torch.zeros(100, 10)
    fake_attention[:, 0] = 1.0
    entropy = model.slot_usage_entropy(fake_attention)
    assert entropy < 0.1

def test_healthy_dictionary_has_higher_entropy():
    model = MotifDictionary(K=10, dim=64, window=3)
    uniform_attention = torch.ones(100, 10) / 10
    entropy = model.slot_usage_entropy(uniform_attention)
    assert entropy > 2.0

def test_a_s_is_exposed_as_output(synthetic_snapshots):
    model = MotifDictionary(K=10, dim=64, window=3)
    z, a, D = model(synthetic_snapshots[:3])
    assert a is not None
    assert a.shape[-1] == 10

def test_random_vs_learned_ablation_hook_exists():
    model_random = MotifDictionary(K=10, dim=64, window=3, mode="random_untrained")
    model_learned = MotifDictionary(K=10, dim=64, window=3, mode="learned")
    assert not torch.allclose(model_random.D, model_learned.D)

def test_reconstruction_target_full_window(synthetic_snapshots):
    model = MotifDictionary(K=10, dim=64, window=3)
    dummy_window = torch.randn(3, 20, 64)
    recon = model.reconstruct(dummy_window)
    assert recon.shape == dummy_window.shape
```

## Module D Tests (`pcdgm/test_module_d_fusion.py`)

```python
import torch
import pytest
from pcdgm.module_d_fusion import GraphFusion

def test_z_per_node_shape_correct(synthetic_snapshots):
    fusion = GraphFusion(out_dim=100)
    H = torch.randn(20, 64)
    z_per_node = torch.randn(20, 64)
    out = fusion(H, z_per_node, synthetic_snapshots[0])
    assert out.shape == (20, 100)
```

## Module E Tests (`pcdgm/test_module_e_predictor.py`)

```python
import torch
from pcdgm.module_e_predictor import ChangePredictor

def test_predicted_positive_rate_not_trivially_zero():
    model = ChangePredictor(dim=100)
    H = torch.randn(20, 100)
    z = torch.randn(20, 64)
    p = model.which(H, H, z)
    assert p.mean() > 0.001

def test_timing_output_uses_interval_not_fake_precision():
    model = ChangePredictor(dim=100)
    H = torch.randn(20, 100)
    z = torch.randn(20, 64)
    interval = model.when_interval(H, H, z, real_hours=[0,5,8,16,23,27,45,45])
    assert interval in range(8)
```

## Training Loop Tests (`training/test_loop.py`)

```python
def test_loss_terms_logged_separately(training_step_output):
    required_keys = {"L_edge", "L_pattern", "L_diversity", "L_timing", "L_total"}
    assert required_keys.issubset(training_step_output.keys())

def test_checkpoint_selection_uses_validation_not_train(training_history):
    assert training_history["checkpoint_metric_source"] == "val"

def test_seed_propagates_to_all_three_init_points(seeded_run_metadata):
    assert seeded_run_metadata["data_shuffle_seed"] == seeded_run_metadata["requested_seed"]
    assert seeded_run_metadata["model_init_seed"] == seeded_run_metadata["requested_seed"]
    assert seeded_run_metadata["dictionary_init_seed"] == seeded_run_metadata["requested_seed"]

def test_five_seeds_minimum(experiment_config):
    assert len(experiment_config["seeds"]) >= 5
```

## Phase 1 Reproduction Gate (`test_phase1_reproduction.py`)

```python
import pytest

def test_devotg_baseline_auc_matches_paper():
    """Must pass before ANY novel module is trusted."""
    pytest.skip("enable once external/devotg is cloned and runnable")
    # reported_auc = run_devotg_baseline_from_their_repo()
    # assert abs(reported_auc - 0.839) < 0.02
```

## Citation Guard

```python
ALLOWED_CITATIONS = {
    "DevoTG": "arXiv:2606.21940",
    "STEP": "arXiv:2603.05874",
    "Witvliet2021": "Nature, 2021",
    "PseudotimeGRN2026": "arXiv:2605.28659",
}

def test_no_uncited_claims_in_draft(draft_text, cited_sources):
    for source in cited_sources:
        assert source in ALLOWED_CITATIONS, f"unverified citation introduced: {source}"
```

---

# PART VI — EVALUATION CRITERIA

| Criterion | Metric | Comparison point |
|---|---|---|
| Trajectory reconstruction (A2) | Spearman rank correlation, relative spacing error vs. real hours | Sanity check before trusting downstream predictions |
| Which-edge prediction | AUROC, AUPRC | Beat Phase 2 plain-TGN and motif-representation baselines |
| Timing prediction | Correct-interval accuracy (primary); fine-grained MAE (illustrative only) | Beat naive "uniform time" baseline |
| Ranking | Precision@K / Recall@K | K = real observed connectivity deltas per interval |
| Pattern module necessity | ΔAUROC(learned) − ΔAUROC(random), ≥5 seeds, paired significance test | Load-bearing for the entire novelty claim |
| Dictionary health | Slot usage entropy | Report regardless of outcome |
| Interpretability | Qualitative match to AVA/AVB/AVE findings | Exploratory, label clearly |

---

# PART VII — WORKING WITH AI ON THIS PROJECT

## The one-shot warning

This book must never be pasted in full and followed by "build this." Reasons: (1) verification between modules is the entire point — Module B trusting a broken Module A defeats the design; (2) several tests explicitly require earlier modules to already work (the injected-motif recovery test needs a real training loop); (3) one-shot generation reliably produces code that runs cleanly while silently failing exactly the checks this book exists to enforce (temporal leakage, dictionary collapse, broadcast confusion) — failures that surface only once outputs are actually inspected.

## The correct workflow

1. Paste the **Strict AI Instructions** (below) at the start of every session.
2. Give AI one module's spec (Part III) + that module's tests (Part V) at a time.
3. AI implements; you run the tests yourself.
4. Fix and iterate until green before moving to the next module.
5. Phase 1 (DevoTG reproduction) must pass before any Part III module is trusted, regardless of how well its own synthetic tests pass.
6. Never let AI draft results-section prose before Phase 6 has actually run.

## Strict AI Instructions (paste verbatim at the start of every coding session)

**MUST DO:**
1. Implement exactly the equations given, using the exact variable names given. If ambiguous, ask — do not silently fill the gap.
2. Temporal splits only. Never `shuffle=True` on this data.
3. Expose every intermediate output named in the spec (`a_s`, `z_s`) as a real, retrievable output — never compute and discard internally.
4. Add the exact assertions/tests listed for this module, as real runtime checks.
5. Log every loss term separately, every step.
6. State explicitly, in a comment, every design choice not fully specified in the prompt, and why.
7. LayerNorm, not BatchNorm, wherever normalization is added.
8. Propagate the seed to data shuffling, model init, AND dictionary init — three separate calls.
9. If a reproduction number (e.g. DevoTG's 0.839) doesn't match, diagnose the actual cause — do not tune hyperparameters until numbers happen to align.
10. State explicitly what splitting/sampling strategy any imported library utility uses internally.

**MUST NOT DO:**
1. Do not add features, modules, or "improvements" not in the spec.
2. Do not treat the two 45h snapshots as sequential timepoints.
3. Do not use activity/calcium-imaging/electrophysiological data in any form.
4. Do not write results-section prose, abstracts, or conclusions asserting the method "works" before Phase 6 has run. Use `[RESULT]` placeholders instead.
5. Do not fabricate or auto-complete citations. Only: DevoTG (arXiv:2606.21940), STEP (arXiv:2603.05874), Witvliet et al. 2021 (*Nature*), the 2026 GRN pseudotime paper (arXiv:2605.28659). Flag anything else as needing verification.
6. Do not silently change the evaluation protocol between baseline and proposed-model runs.
7. Do not initialize the motif dictionary from data statistics unless explicitly running that as a separate ablation variant.
8. Do not report timing precision finer than the real snapshot boundaries allow.
9. Do not select checkpoints using training-set metrics.
10. Do not generate the entire pipeline in one response. One module at a time; stop for review.

If any instruction conflicts with a request in the same prompt, follow these instructions and flag the conflict explicitly.

## Anomaly-to-Test Cross-Reference

| Anomaly | Test that catches it |
|---|---|
| Temporal leakage via random split | `test_no_temporal_leakage` |
| 45h replicates treated as sequential | `test_replicate_snapshots_not_sequential` |
| Neuron ID drift across snapshots | `test_node_id_consistency` |
| BatchNorm instability at small N | `test_no_batchnorm_used` |
| Dictionary collapse | `test_dictionary_collapse_detection` |
| `a_s` discarded internally | `test_a_s_is_exposed_as_output` |
| Random-patterns ablation not truly isolated | `test_random_vs_learned_ablation_hook_exists` |
| Reconstruction collapsed to last timestep | `test_reconstruction_target_full_window` |
| `z_t` broadcast confusion | `test_z_per_node_shape_correct` |
| Model collapses under class imbalance | `test_predicted_positive_rate_not_trivially_zero` |
| Fabricated fine-grained timing | `test_timing_output_uses_interval_not_fake_precision` |
| Loss terms not logged separately | `test_loss_terms_logged_separately` |
| Checkpoint selection on train metrics | `test_checkpoint_selection_uses_validation_not_train` |
| Seed not propagated to all init points | `test_seed_propagates_to_all_three_init_points` |
| Phase 1 baseline unverified | `test_devotg_baseline_auc_matches_paper` |
| Fabricated citation | `test_no_uncited_claims_in_draft` |

**Every row must have a passing test before the corresponding claim appears in the paper.**

---

*End of book. Start at Part IV, Phase 0, whenever ready.*
