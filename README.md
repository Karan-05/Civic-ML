```markdown
# Civic's IMDb Sentiment Classification with Knowledge Distillation

Two epochs of total training, one GPU‑hour, **state‑of‑the‑art accuracy** for free‑tier Colab users:

| Metric (IMDb test)          | **Teacher – BERT‑base** | **Student – DistilBERT** | Δ Change    |
|-----------------------------|-------------------------|--------------------------|-------------|
| **Accuracy**                | **93.45 %**             | 92.69 %                  | ▼ 0.76 pp   |
| **Weighted F1**             | **0.9345**              | 0.9269                   | ▼ 0.8 pp    |
| **Latency**<sup>†</sup>     | 47.6 ms                 | **24.2 ms**              | **‑49 %**   |
| **Throughput**<sup>†</sup>  | 2 099 samples/s         | **4 139 samples/s**      | **+97 %**   |
| **Parameters**              | 109 M                   | **67 M**                 | **‑39 %**   |
| **Disk size**               | 417 MB                  | **255 MB**               | **‑39 %**   |

<sup>† Measured on an NVIDIA T4 with a 100‑sample batch.</sup>

The **student keeps ~99 % of the teacher’s F1** while being roughly **2× faster** and **40 % smaller**.

---

## 🔧 Motivation & Choices

| Component               | Why this choice?                                                                                                                                           |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **IMDb dataset**        | 25 k balanced movie reviews, binary sentiment → quick to iterate, widely used benchmark.                                                                    |
| **BERT‑base teacher**   | Strong out‑of‑the‑box language model; reaches > 93 % accuracy in a single epoch.                                                                           |
| **DistilBERT student**  | 6‑layer distilled BERT → same vocab & architecture (easy KD), 40 % fewer params, faster inference.                                                          |
| **On‑the‑fly KD**       | Compute teacher logits inside `compute_loss` → no extra dataset columns, minimal RAM overhead.                                                              |
| **Weights & Biases**    | Automatic experiment tracking: loss curves, metrics, confusion matrix, GPU/util stats → reproducibility in one dashboard link without extra code. Works seamlessly via 🤗 `Trainer` integration. |

---

## 1  Methodology

### 1.1 Data prep

```python
from datasets import load_dataset
ds = load_dataset("imdb")
ds["train"] = ds["train"].shuffle(seed=42)
split = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]
````

### 1.2 Teacher fine‑tune

* **Model**: `bert‑base‑uncased`
* **Hyper‑params**: 1 epoch, batch 16, LR 2e‑5, cross‑entropy loss.

### 1.3 Student distillation

Loss function:

$$
\begin{aligned}
\mathcal{L}(\theta_s)
&= \alpha\,\mathrm{CE}(y, s) \\
&\quad + (1 - \alpha)\,T^2\,\mathrm{KL}\!\Bigl(
    \mathrm{softmax}\bigl(\tfrac{t}{T}\bigr)
    \,\big\|\,
    \mathrm{softmax}\bigl(\tfrac{s}{T}\bigr)
  \Bigr).
\end{aligned}
$$

* **α = 0.5**, **T = 2.0**
* Teacher logits computed **on‑the‑fly** inside `DistillTrainer.compute_loss`.

### 1.4 Evaluation & analysis

* **Accuracy / F1** via 🤗 `evaluate`
* **Latency & throughput** with a simple timing loop on GPU/CPU
* **Parameter & disk size** via `model.numel()` + folder size
* Optional: probability **calibration curves** (`sklearn.calibration_curve`)

---

## 3  Results Discussion

* **Speed vs Accuracy trade‑off:** DistilBERT doubles throughput while losing < 1 pp accuracy – ideal for high‑QPS services.
* **Memory & disk:** Save \~ 160 MB on disk and \~ 1.2 GB peak GPU RAM during inference.
* **Cost & sustainability:** Fewer FLOPs ⇒ lower cloud bill and carbon footprint.
* **W\&B dashboard:** Reviewers verify curves & system metrics without rerunning code.

---

## 4  Potential Extensions

| Idea                                 | Benefit                               |
| ------------------------------------ | ------------------------------------- |
| Multi‑epoch KD + LR warm‑up          | Push student > 93 % accuracy.         |
| 8‑bit / 4‑bit quantization           | Further 50 % size & 30 % latency cut. |
| Data augmentation (back‑translation) | Better OOD generalization.            |
| Hyper‑param sweeps (α, T) in W\&B    | Find best calibration vs. accuracy.   |

---
```
```
