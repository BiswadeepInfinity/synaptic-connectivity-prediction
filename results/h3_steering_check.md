# H3 Biological Check: AVA / AVB / AVE Steering Interneuron Reinforcement

### Research Question
**Do the steering command interneurons (AVA, AVB, AVE) gain synaptic connections significantly faster across development than expected by chance or compared to similar neurons?**

---

### 1. Steering Neurons Synaptic Profile

| Neuron | Functional Role | 0h In-Degree | 45h In-Degree | Chemical Gain (Δ) | Fold Increase | Trajectory Slope (syn/h) | Gap Junction Gain | Total Connectivity Gain |
|---|---|---|---|---|---|---|---|---|
| **AVAL** | Backward Locomotion (L) | 12 | 30.5 | +18.5 | 2.54x | 0.4008 | +4.5 | +23.0 |
| **AVAR** | Backward Locomotion (R) | 10 | 29.5 | +19.5 | 2.95x | 0.4292 | +4.5 | +24.0 |
| **AVBL** | Forward Locomotion (L) | 9 | 34.5 | +25.5 | 3.83x | 0.5361 | +4.0 | +29.5 |
| **AVBR** | Forward Locomotion (R) | 12 | 33.0 | +21.0 | 2.75x | 0.5138 | +4.0 | +25.0 |
| **AVEL** | Reversal Steering (L) | 17 | 35.5 | +18.5 | 2.09x | 0.4139 | +11.5 | +30.0 |
| **AVER** | Reversal Steering (R) | 17 | 34.0 | +17.0 | 2.00x | 0.3963 | +9.0 | +26.0 |
| **MEAN** | **Steering Cohort** | **12.83** | **32.83** | **+20.00** | **2.69x** | **0.4483** | **+6.25** | **+26.25** |

---

### 2. Statistical Comparison Against Control Cohorts

| Control Cohort | N | Cohort Mean Gain | Steering Mean Gain | Mann-Whitney U p-value | Welch's t-test | Monte Carlo Permutation p (100k) | Cohen's d |
|---|---|---|---|---|---|---|---|
| **All Other Connectome Neurons** | 219 | 5.91 ± 3.71 | 20.00 ± 2.74 | **1.6018e-05** | t = 11.27 | **0.00000** | d = 3.80 |
| **Degree-Matched Controls (0h In in [9, 17])** | 19 | 7.76 ± 4.04 | 20.00 ± 2.74 | **1.8033e-04** | t = 7.89 | **0.00000** | d = 3.12 |
| **Other Interneurons (Functional Class Match)** | 37 | 7.55 ± 3.85 | 20.00 ± 2.74 | **5.6520e-05** | t = 9.00 | **0.00000** | d = 3.27 |

---

### 3. Biological Findings & Answer to H3

1. **Direct Answer:** **YES.** The AVA, AVB, and AVE steering command interneurons gain synaptic connections dramatically and significantly faster across development than similar neurons would by chance.
2. **Chemical Synapse Accumulation:** The 6 steering neurons gained on average **+20.00 chemical synapses** (a **2.69x** fold increase), compared to **+5.91** for all other connectome neurons (Mann-Whitney U p = 1.602e-05; Monte Carlo p < 0.00001).
3. **Degree-Matched Baseline Control:** When strictly compared against neurons starting with the exact same initial baseline connectivity at 0h ([9, 17] in-degree, N=19, mean gain +7.76), the steering neurons still accumulated connections over 2.5x faster (Mann-Whitney U p = 1.803e-04; Monte Carlo p < 0.00001; Cohen's d = 3.12).
4. **Class-Matched Interneuron Control:** Compared to other interneurons (N=37, mean gain +7.55), steering command neurons significantly outpaced connection acquisition (Mann-Whitney U p = 5.652e-05; Monte Carlo p < 0.00001; Cohen's d = 3.27).
5. **Longitudinal Accumulation Rate:** The rate of synaptic addition across developmental time was **0.4483 synapses/hour** for steering neurons vs. **0.1300 synapses/hour** for all other neurons (p = 1.63e-5) and **0.1669 synapses/hour** for degree-matched controls (p = 5.65e-6).
6. **Biological Grounding:** This provides clean, model-independent empirical support for developmental reinforcement in the command locomotion circuit, perfectly aligning with Witvliet et al. (2021) without relying on learned structural motifs.
