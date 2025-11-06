# Activation Function Ablation Study for 3D Point Cloud Classification

**Tagline:** ReLU vs Leaky ReLU vs GELU in a PointNet baseline — accuracy, stability, and latency under a fair, single-switch setup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/158JWPHH55_zEYLZ2anqLM5NEZWqmfe_3?usp=sharing)

## TL;DR

* **Leaky ReLU and GELU tied** for best accuracy (89.98%) on ModelNet10 in this setup.
* **ReLU** delivered the fastest inference (2.4× faster than GELU) and is the most deployment/INT8-friendly.
* GELU trained stably but was significantly slower on this backbone.

---

## Objective

Quantify the impact of **ReLU**, **Leaky ReLU (α = 0.01)**, and **GELU (tanh approximation)** on **accuracy**, **training stability**, and **inference latency** in a PointNet classifier trained on **ModelNet10**.

---

## Results Summary

| Activation | Best Acc (%) | Latency (ms/batch) | Latency (ms/sample) | Speed vs ReLU |
| :--------- | -----------: | :----------------- | ------------------: | :------------ |
| **RELU**   |        89.76 | **28.64 ± 1.99**   |           **0.224** | **1.00×**     |
| **LEAKY**  |    **89.98** | 39.82 ± 2.17       |               0.311 | 0.72×         |
| **GELU**   |    **89.98** | 69.86 ± 1.66       |               0.546 | 0.41×         |

Key takeaways:

* Accuracy: **Leaky ReLU and GELU tied** at 89.98% (+0.22 pp vs ReLU).
* Speed winner: **ReLU** (2.4× faster than GELU per batch on a T4).
* Stability: All three converge smoothly; GELU shows slightly lower final gradient norms (0.67 vs 0.74-0.77).

---

## Experimental Setup

* **Model:** Simplified **PointNet** with a single activation switch (applied identically in backbone and head).
* **Dataset:** **ModelNet10** (~5k CAD meshes), 1,024 points per shape (surface sampling, normalized).
* **Training:** **10 epochs**, Adam (lr = 0.001), **batch size 128**, StepLR at epoch 20 (γ = 0.5).
* **Initialization:** **He/Kaiming** for all runs.
* **Seed:** 42 (single-seed; see Limitations).
* **Hardware:** Google Colab, Tesla T4.

Fair-comparison guarantees:

* Same activation everywhere (no mixing).
* Same initialization, seed, optimizer, schedule, and augmentations.
* Latency measured with warmup, `model.eval()`, and CUDA synchronization; both per-batch and per-sample reported.
* Dead units tracked **per-channel** (a channel counted "dead" if it never activates on the evaluation set), not element-wise sparsity.

---

## Visualizations

<p align="center">
  <img src="figs/best_test_acc.png" alt="Best Test Accuracy" width="420">
  <img src="figs/latency_batch.png" alt="Inference Latency (ms/batch)" width="420">
</p>
<p align="center">
  <img src="figs/speed_vs_accuracy.png" alt="Speed vs Accuracy trade-off" width="420">
  <img src="figs/gradient_norms.png" alt="Gradient Norms (training stability)" width="420">
</p>

---

## Findings

1. **Accuracy**  
   Leaky ReLU and GELU both achieved **89.98%**, tying for best accuracy. ReLU reached **89.76%** (only 0.22 pp behind). The differences are within typical single-seed variance, suggesting **all three activations perform comparably** on this task.

2. **Latency**  
   ReLU is consistently fastest and fuses well in typical deployment stacks. GELU's tanh/erf math introduces **2.4× overhead** vs ReLU. Leaky adds **39% overhead** vs ReLU.

3. **Training stability**  
   All three converge smoothly. Final gradient norms: ReLU 0.74, Leaky 0.77, GELU **0.67** (lowest, indicating smoother optimization).

4. **Dead units**  
   With proper initialization and normalization, dead ReLUs were **negligible: 0.0%** of channels never activated on the eval set. Leaky's benefit is mainly insurance under tougher conditions (e.g., poorer inits, stronger imbalance).

---

## When to Use Which Activation

### ReLU

* **Choose for:** tight latency budgets, embedded/edge deployment, and INT8 quantization.
* **Why:** fastest (28.64 ms/batch), widely optimized, quantization-friendly.
* **This study:** **89.76%** accuracy, 0.224 ms/sample, **0.0% dead neurons**.

### Leaky ReLU (α = 0.01)

* **Choose for:** slight robustness boost against dead units or mild class imbalance, with a modest latency cost.
* **This study:** **89.98%** accuracy (tied best), 39.82 ms/batch, **+39% latency** vs ReLU.

### GELU (tanh approximation)

* **Choose for:** attention/Transformer-style blocks or tasks where smooth gating helps and latency is less critical.
* **Caveat:** typically less INT8-friendly without QAT or special kernels.
* **This study:** **89.98%** accuracy (tied best), 69.86 ms/batch, **+144% latency** vs ReLU.

---

## Key Insight: Marginal Differences

With only **0.22%** accuracy difference between ReLU (89.76%) and the tied leaders (89.98%), the **practical choice depends on your constraints**:

* **Latency-critical (real-time)?** → **ReLU** (2.4× faster than GELU, 0.0% dead neurons with proper init)
* **Slight accuracy/robustness boost acceptable?** → **Leaky ReLU** (39% slower but tied for best)
* **Transformer/attention modules?** → **GELU** (smooth gradients, but 2.4× slower)

---

## Reproducibility

* Single-notebook pipeline (Colab) with sections: setup → data → model (activation switch) → training → evaluation → profiling → plots.
* Save checkpoints and logs after each run.
* Warm up before timing; report both ms/batch and ms/sample with the same batch size across runs.

---

## Limitations

* **CAD vs LiDAR:** ModelNet10 meshes are CAD; results generalize directionally, not absolutely, to LiDAR perception.
* **Single seed:** Numbers can shift slightly; run three seeds for tighter confidence intervals.
* **Short training (10 epochs):** Longer training may increase absolute accuracy but is unlikely to change relative ranking.
* **Batch size 128:** Latency numbers scale with batch size; measure on target hardware for final decisions.

---

## Conclusion

On this PointNet baseline and setup: **Leaky ReLU and GELU tied for best accuracy** (89.98%), **ReLU delivered the best latency** (28.64 ms/batch) and deployability, and all three trained stably. 

**For real-time perception stacks:** Start with **ReLU** (fastest, 0.0% dead neurons with He init). Consider **Leaky** if you observe dead-unit issues or class imbalance in your specific dataset. Reserve **GELU** for attention-heavy modules where its smooth gradients provide tangible benefits and latency budgets allow.

**Practical recommendation:** Given the **marginal 0.22% accuracy difference** and **2.4× speed advantage**, **ReLU is the pragmatic choice** for production PointNet-style models unless your application specifically demands the robustness of Leaky or smoothness of GELU.
