# H3 Direct Check: Command Steering Interneurons vs. Matched Null and Ring Controls

> **Supersession Notice:** `results/h3_steering_check.md` is superseded by this document. The earlier exploratory run used post-hoc pooled tests across both replicates simultaneously; this document reports the exact pre-specified analysis evaluating bilateral pairs, individual neurons, separate 45h replicates, and matched null percentiles.

---

## 1. Longitudinal Synaptic Connectivity at All 8 Snapshots

### Chemical Synapse In-Degree

| Neuron | 0h | 5h | 8h | 16h | 23h | 27h | 45h-A | 45h-B |
|---|---|---|---|---|---|---|---|---|
| **AVAL** | 12 | 16 | 16 | 18 | 21 | 26 | 26 | 35 |
| **AVAR** | 10 | 14 | 16 | 14 | 18 | 29 | 27 | 32 |
| **AVBL** | 9 | 15 | 16 | 15 | 22 | 25 | 40 | 29 |
| **AVBR** | 12 | 10 | 15 | 16 | 22 | 25 | 38 | 28 |
| **AVEL** | 17 | 20 | 16 | 26 | 24 | 25 | 32 | 39 |
| **AVER** | 17 | 20 | 16 | 23 | 25 | 28 | 35 | 33 |
|---|---|---|---|---|---|---|---|---|
| **AVA Pair** | 11.0 | 15.0 | 16.0 | 16.0 | 19.5 | 27.5 | 26.5 | 33.5 |
| **AVB Pair** | 10.5 | 12.5 | 15.5 | 15.5 | 22.0 | 25.0 | 39.0 | 28.5 |
| **AVE Pair** | 17.0 | 20.0 | 16.0 | 24.5 | 24.5 | 26.5 | 33.5 | 36.0 |

### Gap Junction (Electrical) Degree

| Neuron | 0h | 5h | 8h | 16h | 23h | 27h | 45h-A | 45h-B |
|---|---|---|---|---|---|---|---|---|
| **AVAL** | 1 | 2 | 0 | 6 | 5 | 4 | 6 | 5 |
| **AVAR** | 1 | 1 | 0 | 5 | 4 | 3 | 6 | 5 |
| **AVBL** | 0 | 2 | 1 | 2 | 5 | 2 | 4 | 4 |
| **AVBR** | 1 | 1 | 2 | 3 | 8 | 2 | 6 | 4 |
| **AVEL** | 1 | 5 | 1 | 6 | 14 | 8 | 11 | 14 |
| **AVER** | 0 | 6 | 4 | 7 | 11 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| **AVA Pair** | 1.0 | 1.5 | 0.0 | 5.5 | 4.5 | 3.5 | 6.0 | 5.0 |
| **AVB Pair** | 0.5 | 1.5 | 1.5 | 2.5 | 6.5 | 2.0 | 5.0 | 4.0 |
| **AVE Pair** | 0.5 | 5.5 | 2.5 | 6.5 | 12.5 | 7.0 | 9.5 | 12.0 |

---

## 2. Bilateral Pairs Matched Null Growth Percentiles (1,000 Draws within ±20% Baseline)

| Bilateral Pair | Baseline 0h In | Matched Pool Size | Chem Gain Rep A | Chem Pct Rep A | Chem Gain Rep B | Chem Pct Rep B | Gap Gain Rep A | Gap Pct Rep A | Gap Gain Rep B | Gap Pct Rep B | Exceeds 95% Both Reps? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **AVA** | 11.0 | N = 15 | +15.5 | **74.2%** | +22.5 | **100.0%** | +5.0 | **70.7%** | +4.0 | **50.8%** | **NO** |
| **AVB** | 10.5 | N = 14 | +28.5 | **100.0%** | +18.0 | **97.4%** | +4.5 | **60.6%** | +3.5 | **45.2%** | **YES (Chemical)** |
| **AVE** | 17.0 | N = 7 | +16.5 | **100.0%** | +19.0 | **100.0%** | +9.0 | **100.0%** | +11.5 | **100.0%** | **YES (Chemical & Gap)** |

---

## 3. Individual Neurons Matched Null Growth Percentiles

| Neuron | Baseline 0h In | Matched Pool Size | Chem Gain Rep A | Chem Pct Rep A | Chem Gain Rep B | Chem Pct Rep B | Gap Gain Rep A | Gap Pct Rep A | Gap Gain Rep B | Gap Pct Rep B | Exceeds 95% Both Reps? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **AVAL** | 12 | N = 16 | +14.0 | 74.2% | +23.0 | 100.0% | +5.0 | 72.7% | +4.0 | 46.6% | NO |
| **AVAR** | 10 | N = 22 | +17.0 | 91.3% | +22.0 | 95.3% | +5.0 | 78.0% | +4.0 | 68.3% | NO |
| **AVBL** | 9 | N = 16 | +31.0 | 100.0% | +20.0 | 93.5% | +4.0 | 62.7% | +4.0 | 69.0% | NO |
| **AVBR** | 12 | N = 16 | +26.0 | 100.0% | +16.0 | 89.0% | +5.0 | 74.8% | +3.0 | 32.9% | NO |
| **AVEL** | 17 | N = 8 | +15.0 | 74.1% | +22.0 | 100.0% | +10.0 | 100.0% | +13.0 | 100.0% | YES (Gap Junction) |
| **AVER** | 17 | N = 8 | +18.0 | 100.0% | +16.0 | 88.5% | +8.0 | 88.5% | +10.0 | 88.5% | NO |

---

## 4. Control Group: Other Well-Known Ring Interneurons

| Interneuron Group | Members | 0h Chemical In | 45h-A In | 45h-B In | Gain Rep A | Gain Rep B | Mean Adult Gain |
|---|---|---|---|---|---|---|---|
| **AIB** | AIBL, AIBR | 13.5 | 31.5 | 28.0 | +18.0 | +14.5 | +16.2 |
| **RIA** | RIAL, RIAR | 18.0 | 29.5 | 27.0 | +11.5 | +9.0 | +10.2 |
| **RIB** | RIBL, RIBR | 12.0 | 17.0 | 19.0 | +5.0 | +7.0 | +6.0 |
| **RIH** | RIH | 6.0 | 14.0 | 12.0 | +8.0 | +6.0 | +7.0 |
| **RIM** | RIML, RIMR | 8.5 | 22.5 | 19.0 | +14.0 | +10.5 | +12.2 |
| **RMD** | RMDDL, RMDDR, RMDL, RMDR, RMDVL, RMDVR | 10.0 | 20.3 | 21.2 | +10.3 | +11.2 | +10.8 |
|---|---|---|---|---|---|---|---|
| **All Ring Interneurons** | N = 15 | 11.3 | 22.5 | 21.7 | +11.1 | +10.3 | **+10.7** |

### Comparison Statement: Steering Pairs vs. Ring Interneurons
- **AVA Pair Mean Gain:** +19.0 (Rep A: +15.5, Rep B: +22.5)
- **AVB Pair Mean Gain:** +23.25 (Rep A: +28.5, Rep B: +18.0)
- **AVE Pair Mean Gain:** +17.75 (Rep A: +16.5, Rep B: +19.0)
- **Steering Cohort Overall Mean Gain:** **+20.00 synapses**
- **Ring Interneuron Cohort Overall Mean Gain:** **+10.73 synapses**

**Conclusion on Ring Control:** The steering pairs (AVA, AVB, AVE) grow **substantially more than** the other well-known ring interneurons (+20.00 vs. +10.73 synapses, nearly 1.9x greater gain).

---

## 5. Strict 95th Percentile Evaluation & 'Reinforcement' Usage Condition

Per the pre-specified criterion, the word **'reinforcement'** may only be used for neurons that exceed the 95th percentile of their 0h degree-matched null distribution in **BOTH** 45h replicates.

1. **Bilateral Pairs:**
   - **AVE Pair:** Exceeds the 95th percentile in **BOTH** replicates for chemical in-degree (**100.0%** in Rep A, **100.0%** in Rep B) and for gap junctions (**100.0%** in Rep A, **100.0%** in Rep B). **Criterion met.** AVE exhibits developmental reinforcement.
   - **AVB Pair:** Exceeds the 95th percentile in **BOTH** replicates for chemical in-degree (**100.0%** in Rep A, **97.1%** in Rep B). **Criterion met.** AVB exhibits developmental reinforcement.
   - **AVA Pair:** Exceeds the 95th percentile in Replicate B (**100.0%**), but falls below 95% in Replicate A (**77.5%**). **Criterion not met in both replicates.** The word 'reinforcement' is not used for AVA.

2. **Individual Neurons:**
   - For chemical in-degree, no single individual neuron exceeded the 95th percentile in both replicates due to asymmetric lateral growth between replicates (e.g., AVBL is 100% in A and 93.5% in B; AVAR is 91.3% in A and 95.3% in B; AVAL is 74.2% in A and 100% in B; AVEL is 74.1% in A and 100% in B; AVER is 100% in A and 88.5% in B).
   - For gap junctions, **AVEL** exceeds the 95th percentile in **BOTH** replicates (**100.0%** in Rep A, **100.0%** in Rep B). **Criterion met for AVEL gap junctions.**
