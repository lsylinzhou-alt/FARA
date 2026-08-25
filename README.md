<h1 align="center">FARA: Forget-resistant Multi-level Knowledge Distillation for Continual Learning of Vision–Language Models</h1>

<p align="center">
Anonymous Submission
</p>

<p align="center">
    <!-- Replace the links below after the paper/code is publicly available -->
    <a href="#"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Code-Coming%20Soon-blue.svg"></a>
</p>

<p align="center">
    <a href="#overview">Overview</a> |
    <a href="#method">Method</a> |
    <a href="#environment-set-up">Environment Set Up</a> |
    <a href="#prepare-datasets">Prepare Datasets</a> |
    <a href="#training">Training</a> |
    <a href="#results">Results</a> |
    <a href="#citation">Citation</a> |
    <a href="#acknowledgement">Acknowledgement</a>
</p>

---

This repository contains the implementation of **FARA: Forget-resistant Multi-level Knowledge Distillation for Continual Learning of Vision–Language Models**.

Continual adaptation enables vision–language models (VLMs) to continuously acquire knowledge from sequentially arriving domains. However, repeated model updates may lead to **catastrophic forgetting** of previously learned tasks and degradation of the **zero-shot transferability** inherited from large-scale pre-training.

To address these challenges, we propose **FARA**, a forget-resistant multi-level knowledge distillation framework for continual VLM adaptation. Instead of preserving knowledge only at the parameter, logit, or pointwise feature level, FARA protects knowledge from three complementary perspectives:

* **Geometric Feature Distillation (GFD)** preserves the relational topology of teacher representations.
* **Jensen–Shannon Self-Attention Distillation (JS-SAD)** aligns spatial attention distributions between the teacher and student.
* **Feature Space Smoothing and Generalization (FSSG)** regularizes local model responses around intermediate representations.

Together, these components form a unified **geometry–attention–response protection mechanism**, improving the stability–plasticity trade-off during continual learning without introducing additional inference-time parameters.

---

## Overview

<!-- Replace this path with the actual framework figure in your repository -->

![Framework Overview](path/to/framework.png)

**Framework overview of FARA.**

FARA follows a teacher–student continual adaptation paradigm based on synthetic replay.

At task (t), the model learned after task (t-1) is frozen as the **teacher model**, while a copy is optimized as the **student model**. Each mini-batch combines samples from the current task with synthetic replay samples corresponding to previously observed domains.

FARA transfers complementary knowledge at three levels:

1. **Representation Geometry** — GFD preserves pairwise distances and triplet-wise angular relations.
2. **Spatial Attention** — JS-SAD aligns task-relevant visual attention between the frozen teacher and current student.
3. **Local Response Behavior** — FSSG perturbs intermediate representations and encourages teacher and student networks to respond consistently to the same local perturbations.

The overall objective is

[
\mathcal{L}_{total}
===================

\mathcal{L}*{task}
+
\alpha\mathcal{L}*{GFD}
+
\beta\mathcal{L}*{JS-SAD}
+
\gamma\mathcal{L}*{FSSG}.
]

All auxiliary objectives are used only during training. The original CLIP inference architecture remains unchanged.

---

## Method

### Geometric Feature Distillation

Conventional pointwise feature matching independently constrains individual examples but may still allow the global representation space to deform during continual learning.

To better preserve previously acquired knowledge, **Geometric Feature Distillation (GFD)** transfers relational knowledge from the frozen teacher to the student.

GFD preserves two complementary structures:

* **Distance relations**, which maintain global separation among samples.
* **Angular relations**, which preserve local orientation among sample triplets.

The complete geometric objective is

[
\mathcal{L}_{GFD}
=================

\lambda_d\mathcal{L}*{dist}
+
\lambda_a\mathcal{L}*{angle}.
]

In our experiments, we use

```text
lambda_d = 1
lambda_a = 1
```

while the overall strength of GFD is controlled by (\alpha).

---

### Jensen–Shannon Self-Attention Distillation

Feature geometry alone does not specify which spatial regions contribute to the prediction.

We therefore introduce **Jensen–Shannon Self-Attention Distillation (JS-SAD)** to preserve task-relevant spatial attention.

For selected visual Transformer layers, the class-token-to-patch attention logits are converted into temperature-scaled spatial probability distributions. The student attention distribution is then aligned with that of the frozen teacher using **Jensen–Shannon divergence**.

Compared with asymmetric KL-based alignment, Jensen–Shannon divergence is symmetric and bounded, providing stable optimization when teacher and student attention distributions differ significantly.

---

### Feature Space Smoothing and Generalization

Matching clean teacher and student features alone does not constrain how the two models behave in the local neighborhood of a representation.

**Feature Space Smoothing and Generalization (FSSG)** therefore introduces a shared norm-bounded perturbation into an intermediate visual representation.

The same perturbation direction is applied to both teacher and student features before passing through the remaining nonlinear encoder blocks. FARA then aligns:

* the perturbed semantic embeddings;
* the corresponding local response directions.

This formulation constrains the **local transformation behavior** of the student rather than simply matching noisy endpoints.

The FSSG objective is

[
\mathcal{L}_{FSSG}
==================

\mathcal{L}*{nei}
+
\gamma*{dir}\mathcal{L}_{dir}.
]

We use

```text
gamma_dir = 1
```

in the default configuration.

---

## Environment Set Up

Our implementation is based on **PyTorch** and uses **CLIP** as the base vision–language model.

The experiments reported in the paper were conducted on an **NVIDIA RTX 4090 GPU**.

A recommended environment can be created as follows:

```bash
conda create -n fara python=3.8
conda activate fara
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Please update the Python/PyTorch/CUDA versions above according to the final released implementation.

---

## Prepare Datasets

We evaluate FARA on the **Multi-Domain Task Incremental Learning (MTIL)** benchmark.

The benchmark contains **11 heterogeneous visual recognition datasets with 1,201 classes**:

* Aircraft
* Caltech101
* CIFAR100
* DTD
* EuroSAT
* Flowers102
* Food101
* MNIST
* OxfordPets
* StanfordCars
* SUN397

The datasets arrive sequentially following two predefined task orders:

```text
MTIL Order I
MTIL Order II
```

Please download each dataset from its corresponding official source and organize them according to the dataset configuration used by the project.

```text
data/
├── aircraft/
├── caltech101/
├── cifar100/
├── dtd/
├── eurosat/
├── flowers102/
├── food101/
├── mnist/
├── oxfordpets/
├── stanfordcars/
└── sun397/
```

> Replace the directory structure above if the released implementation uses a different organization.

---

## Synthetic Replay

FARA does **not store real historical samples** during continual learning.

Instead, synthetic replay samples are used to revisit previously acquired domain knowledge. For controlled comparison, FARA follows the same synthetic replay setting as the GIFT baseline, including the pre-trained CLIP backbone, synthetic replay pool, prompts, and replay budget.

During task (t):

```text
Previous checkpoint  →  Frozen Teacher
Current checkpoint   →  Trainable Student

Current-task data
        +
Synthetic replay data
        ↓
Multi-level knowledge distillation
```

The resulting teacher–student framework enables FARA to preserve historical knowledge without requiring storage of the original training images.

---

## Training

Each task is trained for **1,000 iterations** using **AdamW**.

The main experimental configuration reported in the paper is:

```yaml
optimizer: AdamW
iterations_per_task: 1000
batch_size: 64

learning_rate:
  - 1e-5
  - 5e-5

label_smoothing: 0.2

fara:
  alpha_gfd: 100
  beta_js_sad: 0.5
  gamma_fssg: 2.0
  epsilon: 0.1

gfd:
  lambda_distance: 1
  lambda_angle: 1

fssg:
  gamma_direction: 1
```

From the second task onward, the checkpoint obtained from the preceding task is frozen and used as the teacher model.

### Example

After the source code is released, experiments can be launched using commands such as:

```bash
# MTIL Order I
python train.py --config configs/mtil_order_I.yaml

# MTIL Order II
python train.py --config configs/mtil_order_II.yaml
```

> The commands above are placeholders. Please replace them with the actual entry scripts and configuration paths of the released repository.

---

## Results

We evaluate FARA using three metrics commonly adopted in continual VLM learning:

* **Transfer** — evaluates zero-shot accuracy on tasks that have not yet been observed and reflects model plasticity and forward transfer.
* **Average** — summarizes performance throughout the entire task sequence.
* **Last** — reports final average accuracy over previously learned tasks and mainly reflects knowledge retention and stability.

### MTIL Order I

| Method          | Transfer |  Average |     Last |
| --------------- | -------: | -------: | -------: |
| Zero-shot       |     69.4 |     65.3 |     65.3 |
| ZSCL            |     68.1 |     75.4 |     83.6 |
| DIKI            |     68.7 |     76.3 |     85.1 |
| MoE-Adapter     |     68.9 |     76.7 |     85.0 |
| GIFT            |     69.3 |     77.3 |     86.0 |
| IAP             |     69.2 |     76.8 |     85.7 |
| **FARA (Ours)** | **70.1** | **78.1** | **86.6** |

Compared with GIFT, FARA improves:

```text
Transfer : +0.8
Average  : +0.8
Last     : +0.6
```

### MTIL Order II

| Method          | Transfer |  Average |     Last |
| --------------- | -------: | -------: | -------: |
| Zero-shot       |     65.4 |     65.3 |     65.3 |
| ZSCL            |     64.2 |     74.5 |     83.4 |
| DIKI            |     64.4 |     74.5 |     85.5 |
| MoE-Adapter     |     64.3 |     74.7 |     84.1 |
| GIFT            | **65.9** |     75.7 |     85.3 |
| IAP             |     64.9 |     75.1 |     85.9 |
| **FARA (Ours)** | **65.9** | **76.3** | **86.3** |

Under Order II, FARA matches GIFT in Transfer while improving:

```text
Average : +0.6
Last    : +1.0
```

These results indicate that FARA improves historical-task retention without sacrificing forward transfer.

---

## Ablation Study

The three components of FARA provide complementary benefits.

| Method     | Transfer |  Average |     Last |
| ---------- | -------: | -------: | -------: |
| w/o JS-SAD |     69.5 |     77.6 |     85.9 |
| w/o GFD    |     69.8 |     77.8 |     86.5 |
| w/o FSSG   |     69.9 |     77.9 |     85.6 |
| **FARA**   | **70.1** | **78.1** | **86.6** |

Removing **JS-SAD** reduces all three evaluation metrics, demonstrating the value of preserving task-relevant attention.

Removing **GFD** causes smaller but consistent performance degradation, indicating the importance of maintaining relational representation geometry.

Removing **FSSG** results in the largest decrease in final accuracy, highlighting the importance of local response regularization for long-term knowledge retention.

---

## Stability–Plasticity Trade-off

FARA is designed to simultaneously preserve:

```text
Plasticity  → ability to learn and transfer to new domains
Stability   → ability to retain previously acquired knowledge
```

On MTIL Order I, FARA improves both **Transfer** and **Last** compared with GIFT, moving toward the upper-right region of the empirical stability–plasticity space.

This suggests that multi-level knowledge protection can improve retention without excessively restricting adaptation to new domains.

---

## Representation Analysis

After sequential learning of all 11 tasks, the representation-space visualization shows that FARA produces:

* more compact within-task clusters;
* clearer separation between several task groups;
* reduced dense mixing in the central representation region.

These observations are consistent with the intended roles of the three modules:

```text
GFD      → preserve relational geometry
JS-SAD   → preserve task-relevant visual attention
FSSG     → suppress irregular local representation drift
```

---

## Cross-domain Analysis

Across the complete **11 × 11 MTIL training–evaluation matrix**, FARA achieves positive gains over GIFT on **92 of 121 source–target pairs**.

The average improvement is approximately:

```text
+0.92 percentage points
```

The broadly distributed gains suggest that the improvement is not dominated by only a small number of datasets.

FARA still exhibits negative transfer under several severe domain shifts, indicating an interesting direction for future work such as domain-aware loss weighting and adaptive perturbation-layer selection.

---

## Computational Cost

All FARA modules are used only during training.

* GFD uses vectorized pair relations and sampled triplets.
* JS-SAD reuses visual-backbone attention maps.
* FSSG introduces one additional perturbed encoder-tail forward pass.

FARA:

```text
Additional inference parameters : 0
Inference architecture change   : No
Inference latency increase      : No
Approx. training-time increase  : ~5%
```

Therefore, the original CLIP inference path is fully preserved.

---

## Citation

The paper is currently under anonymous review. Citation information will be updated after publication.

```bibtex
@inproceedings{fara,
  title     = {FARA: Forget-resistant Multi-level Knowledge Distillation for Continual Learning of Vision--Language Models},
  author    = {Anonymous},
  booktitle = {To appear},
  year      = {To appear}
}
```

> Please replace the anonymous citation above with the official BibTeX entry after acceptance/publication.

---

## Acknowledgement

FARA builds upon prior research in continual learning, vision–language model adaptation, knowledge distillation, and synthetic replay.

In particular, our experimental framework follows the MTIL continual-learning protocol and uses **GIFT** as an important synthetic-replay baseline.

We sincerely thank the authors of the related open-source projects and datasets for making their work available to the research community.

---

## TODO

* [ ] Release source code
* [ ] Add official paper / arXiv link
* [ ] Add framework figure
* [ ] Add environment requirements
* [ ] Add dataset preparation scripts
* [ ] Add synthetic-data generation instructions
* [ ] Add MTIL Order I / II training commands
* [ ] Add pretrained checkpoints
* [ ] Add evaluation scripts
* [ ] Update official citation
