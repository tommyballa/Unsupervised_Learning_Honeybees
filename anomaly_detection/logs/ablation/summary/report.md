# Ablation: training-set size and backbone

## Protocol

The ablations cover eight models: PatchCore, CFA, Dinomaly, PaDiM, STFPM, FastFlow, PaSTe and RD4AD.
The dataset is the final one (SAM3, batches 3–6) and the protocol is the same as in the main experiments:

- the best checkpoint is chosen on the validation set (322 bees with varroa and 1251 healthy);
- the metrics are computed only once on the held-out test (957 bees with varroa and 5000 healthy);
- maps and masks are at 512×512.

Each ablation changes a single factor. The rest of each model's recipe is unchanged: learning rate, number of steps, layers at the same stride, map post-processing. No re-optimisation was done and there is a single seed per configuration.

- **Number of training bees.** Random subsets of N ∈ {500, 5 000, 20 000} healthy bees are used, in addition to the full training set of 102 303 bees.
  - The subsets are nested (500 ⊂ 5 000 ⊂ 20 000) and obtained from a fixed permutation.
  - For CFA, the number of patches sampled per image when building the memory bank grows as N decreases, so as to keep about 1.6 M candidate patches for clustering.
- **Backbone.** With the full training set, three ImageNet-pretrained CNN backbones are compared: WideResNet-50-2, ResNet-18 and MobileNetV2.
  - For MobileNetV2 the `features.3/6/13` blocks are used, which have the same stride as ResNet layers 1/2/3.
  - FastFlow and RD4AD were tried only with ResNet-18 and WideResNet-50.
  - For Dinomaly, ViT-B/14 is compared with ViT-S/14, both DINOv2 with registers.
- **Inference cost.** It is measured on the same computation as the evaluation: model in fp32, post-processing and map resized back to 512×512. The quantities measured are:
  - **GFLOP per image**, with `torch.utils.flop_counter`. PatchCore's k-NN search (`torch.cdist`) is not seen by the counter and is added as 2·n·m·d.
  - **Parameters**: all tensors of the model, including memory bank and covariance matrices.
  - **Latency at batch 1**: median on an NVIDIA GPU and on CPU with 4 threads.

  The GPUs were shared with other users' jobs, so the latencies should be read as indicative; the main comparison is based on GFLOP.

## Results: number of training bees

Table 1 – Image AUROC / pixel AP on the held-out test as N varies (recipe backbone).

| Model | N = 500 | N = 5 000 | N = 20 000 | N = 102 303 |
|---|---|---|---|---|
| PatchCore | 0.987 / 0.608 | 0.997 / 0.736 | 0.998 / 0.762 | 0.999 / 0.777 |
| CFA | 0.985 / 0.586 | 0.995 / 0.739 | 0.996 / 0.749 | 0.996 / 0.755 |
| Dinomaly | 0.991 / 0.731 | 0.995 / 0.763 | 0.996 / 0.762 | 0.996 / 0.768 |
| PaDiM | 0.986 / 0.480 | 0.989 / 0.518 | 0.989 / 0.522 | 0.989 / 0.521 |
| STFPM | 0.985 / 0.554 | 0.984 / 0.562 | 0.984 / 0.558 | 0.983 / 0.534 |
| FastFlow | 0.982 / 0.441 | 0.975 / 0.423 | 0.982 / 0.482 | 0.982 / 0.469 |
| PaSTe | 0.963 / 0.373 | 0.966 / 0.398 | 0.972 / 0.433 | 0.965 / 0.393 |
| RD4AD | 0.964 / 0.469 | 0.963 / 0.431 | 0.964 / 0.399 | 0.953 / 0.349 |


Figure: `train_size.pdf`.

Memory-based models are the ones most affected by the number of normal bees available: PatchCore, CFA and PaDiM directly model the distribution of normal features.

- **With 500 bees** PatchCore and CFA lose about one point of AUROC and about 17 points of pixel AP compared with full training.
- **Already with 5 000 bees** they recover almost the whole gap. Beyond 20 000 bees the gain is marginal.
- **Dinomaly is the most robust with little data:** with 500 bees it still reaches AUROC 0.991 and pixel AP 0.731.

The models trained by gradient descent (STFPM, FastFlow, PaSTe, RD4AD) have a number of steps fixed by the recipe. In practice they see a few thousand images even when the training set is complete, and indeed their performance does not depend systematically on N: the differences are within run-to-run variability.

RD4AD is an exception: it gets worse as N grows, especially in pixel AP, which drops from 0.469 to 0.349. This is consistent with a recipe optimised for full training that does not benefit from the greater variety, but with a single seed the result should be taken with caution.

From an application point of view, a few thousand healthy bees are enough to install the system on a new hive without significant losses.

## Results: backbone and inference cost

Table 2 – Full training, alternative backbones (★ = recipe backbone). Indicative latencies (shared GPU).

| Model | Backbone | img AUROC | pxl AP | Params (M) | GFLOP | GPU bs1 (ms) | CPU (ms) |
|---|---|---|---|---|---|---|---|
| PatchCore | WRN50 ★ | 0.999 | 0.777 | 71.0 | 476 | 48 | 7590 |
| | R18 | 0.998 | 0.735 | 14.3 | 111 | 20 | 3040 |
| | MNv2 | 0.999 | 0.754 | 4.4 | 36 | 15 | 1460 |
| CFA | WRN50 ★ | 0.996 | 0.755 | 52.4 | 392 | 54 | 1990 |
| | R18 | 0.980 | 0.571 | 9.2 | 73 | 33 | 1510 |
| | MNv2 | 0.991 | 0.553 | 2.7 | 18 | 32 | 1430 |
| Dinomaly | ViT-B ★ | 0.996 | 0.768 | 148.0 | 424 | 59 | 1350 |
| | ViT-S | 0.996 | 0.678 | 37.4 | 121 | 21 | 640 |
| PaDiM | WRN50 ★ | 0.989 | 0.521 | 27.0 | 99 | 11 | 420 |
| | R18 | 0.641 | 0.305 | 2.9 | 15 | 3 | 290 |
| | MNv2 | 0.993 | 0.586 | 0.6 | 2.8 | 5 | 340 |
| STFPM | R18 ★ | 0.983 | 0.534 | 5.6 | 32 | 6 | 190 |
| | WRN50 | 0.985 | 0.568 | 49.8 | 194 | 18 | 960 |
| | MNv2 | 0.940 | 0.371 | 1.1 | 6.6 | 9 | 250 |
| FastFlow | WRN50 ★ | 0.982 | 0.469 | 102.9 | 341 | 49 | 1700 |
| | R18 | 0.913 | 0.304 | 10.4 | 30 | 14 | 470 |
| PaSTe | R18 ★ | 0.965 | 0.393 | 5.6 | 37 | 13 | 630 |
| | WRN50 | 0.916 | 0.196 | 49.8 | 199 | 27 | 1210 |
| | MNv2 | 0.795 | 0.092 | 1.1 | 13 | 16 | 570 |
| RD4AD | WRN50 ★ | 0.953 | 0.349 | 161.2 | 269 | 24 | 1000 |
| | R18 | 0.921 | 0.266 | 27.6 | 41 | 8 | 270 |

Figure: `efficiency.pdf`, which plots image AUROC and pixel AP against GFLOP, on a logarithmic scale.

**Lightweight backbones.** For image-level detection a lightweight backbone is often sufficient; localisation, measured with pixel AP, is more sensitive to feature quality.

- **PatchCore** with MobileNetV2 has the same AUROC as WideResNet-50 (0.999) and a slightly lower pixel AP (0.754 vs 0.777), with 13 times fewer operations (36 vs 476 GFLOP) and 16 times fewer parameters. It is the best trade-off between accuracy and cost and the natural candidate for deployment on an edge device.
- **PaDiM** with MobileNetV2 even beats the recipe backbone, with AUROC 0.993 and only 2.8 GFLOP.
- **Dinomaly** with ViT-S keeps the AUROC but loses 9 points of pixel AP.

**Where the cost is concentrated.** In memory-based methods such as PatchCore, much of the cost is not in the backbone but in the k-NN search: with WideResNet-50 it accounts for 377 of the 476 GFLOP. Reducing the feature dimension with a smaller backbone therefore also lowers the cost of the search.

**Trained models.** The behaviour is less regular, because learning rate and number of steps were optimised for the recipe backbone:

- FastFlow and RD4AD lose a lot when switching to ResNet-18;
- for PaSTe, optimised on ResNet-18, both WideResNet-50 and MobileNetV2 are worse.

Two negative cases should be read with this caveat in mind:

- **PaSTe with MobileNetV2** (AUROC 0.795) has its best checkpoint at the last step, a sign that training did not converge with the ResNet-18 recipe.
- **PaDiM with ResNet-18** (AUROC 0.641) depends on the layer3 features combined with a shared Gaussian. Even when choosing the best covariance regularisation on the validation set, the validation AUROC does not exceed 0.82. The problem is therefore tied to the features and not to an implementation error.

## Limitations

- There is a single seed per configuration, so differences below about 0.005 AUROC are not significant.
- The recipes were not re-optimised for each N and each backbone. The results measure the robustness of the recipe, not the maximum achievable by each combination.
- Latencies were measured on shared GPUs and CPUs; GFLOP is the load-independent cost measure.

## Files

- `logs/ablation/summary/ablation.csv`: all metrics, i.e. image AUROC, AP and F1 and pixel AP, AUPRO and F1, together with the cost measures.
- `logs/ablation/summary/train_size.pdf` and `efficiency.pdf`: the figures.
- Code:
  - `experiments/single_bee/ablation.py`, with the variables `ABL_TRAIN_N` and `ABL_BACKBONE`;
  - `measure_efficiency.py`;
  - `summarize_ablation.py`.
