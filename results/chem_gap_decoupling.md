# Chemical vs Gap-Junction Growth Decoupling Analysis

> 224 neurons analysed (0h chemical in-degree matched pool >= 5). Gain = mean of 45h-A and 45h-B minus 0h. Seed 42.

---

## 1. Correlation of Chemical and Gap-Junction Growth

| Metric | Pearson r | p-value | Spearman r | p-value |
|---|---|---|---|---|
| Raw gain (synapses) | +0.4652 | 0.0000 | +0.4641 | 0.0000 |
| Null percentile | +0.3132 | 0.0000 | +0.3242 | 0.0000 |

**Interpretation:** **Moderate correlation.** There is partial overlap between chemical and gap-junction growth, but the two synapse types are not co-regulated.

---

## 2. Quadrant Analysis (split at population medians)

Chemical gain median: 5.5 synapses | Gap-junction gain median: 1.0

| Quadrant | Count | % of total |
|---|---|---|
| High chemical & high gap-junction | 69 | 30.8% |
| High chemical only | 37 | 16.5% |
| High gap-junction only | 41 | 18.3% |
| Low both | 77 | 34.4% |

---

## 3. Position of H3 Hits in 2D Growth Space

| Neuron | Source | Chem Gain | Gap Gain | Quadrant |
|---|---|---|---|---|
| **AQR** | Gap hit | +10.0 | +5.0 | High-both |
| **AVAL** | H3-direct | +18.5 | +4.5 | High-both |
| **AVAR** | H3-direct | +19.5 | +4.5 | High-both |
| **AVBL** | H3-direct | +25.5 | +4.0 | High-both |
| **AVBR** | H3-direct | +21.0 | +4.0 | High-both |
| **AVEL** | Gap hit | +18.5 | +11.5 | High-both |
| **AVER** | H3-direct | +17.0 | +9.0 | High-both |
| **AVJL** | Chem hit | +14.0 | +1.5 | High-both |
| **AVJR** | Chem hit | +12.0 | +1.5 | High-both |
| **AVKL** | Gap hit | +8.5 | +7.5 | High-both |
| **AVKR** | Gap hit | +8.0 | +7.0 | High-both |
| **AVM** | Gap hit | +7.0 | +4.0 | High-both |
| **OLQDR** | Gap hit | +3.5 | +7.0 | Gap-only |
| **RIBR** | Gap hit | +8.0 | +11.0 | High-both |
| **RIGL** | Gap hit | +8.0 | +7.5 | High-both |
| **RIGR** | Gap hit | +11.5 | +7.5 | High-both |
| **RMDR** | Chem hit | +18.5 | +5.0 | High-both |
| **RMFR** | Chem hit | +15.0 | +2.0 | High-both |
| **RMHL** | Chem hit | +14.0 | +2.0 | High-both |
| **RMHR** | Chem+Gap hit | +15.5 | +3.5 | High-both |
| **SIBVR** | Gap hit | +5.5 | +4.5 | Gap-only |

---

## 4. Top 15 Neurons by Chemical Gain

| Rank | Neuron | Chem Gain | Gap Gain |
|---|---|---|---|
| 1 | AVBL | +25.5 | +4.0 |
| 2 | AVBR | +21.0 | +4.0 |
| 3 | AVAR | +19.5 | +4.5 |
| 4 | RMDR | +18.5 | +5.0 |
| 5 | AVEL | +18.5 | +11.5 |
| 6 | AVAL | +18.5 | +4.5 |
| 7 | AVER | +17.0 | +9.0 |
| 8 | RMDL | +17.0 | +6.0 |
| 9 | AIBR | +17.0 | +5.0 |
| 10 | RMFL | +16.5 | +1.5 |
| 11 | AIBL | +15.5 | +4.0 |
| 12 | RMHR | +15.5 | +3.5 |
| 13 | RMFR | +15.0 | +2.0 |
| 14 | SMDVL | +14.5 | +3.0 |
| 15 | RMHL | +14.0 | +2.0 |

## 5. Top 15 Neurons by Gap-Junction Gain

| Rank | Neuron | Gap Gain | Chem Gain |
|---|---|---|---|
| 1 | AVEL | +11.5 | +18.5 |
| 2 | RIBR | +11.0 | +8.0 |
| 3 | AVER | +9.0 | +17.0 |
| 4 | RMDVR | +8.5 | +6.0 |
| 5 | RMDDL | +7.5 | +9.0 |
| 6 | RIGR | +7.5 | +11.5 |
| 7 | RMDVL | +7.5 | +4.5 |
| 8 | RIGL | +7.5 | +8.0 |
| 9 | AVKL | +7.5 | +8.5 |
| 10 | AVKR | +7.0 | +8.0 |
| 11 | OLQDR | +7.0 | +3.5 |
| 12 | AIZR | +6.5 | +7.0 |
| 13 | RIBL | +6.5 | +4.0 |
| 14 | RIML | +6.5 | +13.0 |
| 15 | RMDDR | +6.0 | +9.5 |
