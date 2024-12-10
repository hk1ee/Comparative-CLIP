# How to reproduce the results

## Steps to Reproduce
1. Download and set up precomputed image features:
   - Download the features from [here](https://huggingface.co/hk1ee/Comparative-CLIP/tree/main)
   - Place the downloaded `precomputed_image_features` directory in the root directory of this project

2. Run experiment scripts:
   ```bash
   cd scripts/experiments
   # Reproduce Table 2
   ./Experiment-ours.sh
   ./Experiment-baselines.sh
   # Reproduce Table 3
   ./Experiment-using_only_descriptor.sh
   # Reproduce Table 4
   ./Experiment-equal_number.sh
   # Reproduce Table 6
   ./Experiment-filter_existing_method.sh
   ```

- The results will be saved as `.csv` files in the `results` directory.

# Baselines
Our code is built upon the following repositories:
- **DCLIP**: *Visual Classification via Description from Large Language Models* (ICLR 2023)  
  [Paper](https://openreview.net/references/pdf?id=MKWeXr89SU) | [GitHub](https://github.com/sachit-menon/classify_by_description_release)
- **WaffleCLIP**: *Waffling around for Performance: Visual Classification with Random Words and Broad Concepts* (ICCV 2023)  
  [Paper](https://arxiv.org/pdf/2306.07282.pdf) | [GitHub](https://github.com/ExplainableML/WaffleCLIP?tab=readme-ov-file)

# Results

|     ViT-B/32     |  IN1k | IN1k-V2 | Caltech | CIFAR |  CUB  |  SAT  | Places |  Food |  Pets |  DTD  | Flowers | Aircraft |  Cars |  Avg  |
|:----------------:|:-----:|:-------:|:-------:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:-------:|:--------:|:-----:|:-----:|
| CLIP             | 62.04 |  54.67  |  78.39  | 64.31 | 51.36 | 40.89 |  39.11 | 82.57 | 85.04 | 43.19 |  62.97  |   24.96  | 58.62 | 57.55 |
| DCLIP            | 63.66 |  56.28  |  81.23  | 64.94 | 53.61 | 41.37 |  41.64 | 83.06 | 85.25 | 44.26 |  66.63  |   26.67  | 59.08 | 59.05 |
| WaffleCLIP       | 63.32 |  55.97  |  81.06  | 65.44 | 52.47 | 43.66 |  40.67 | 82.84 | 85.48 | 42.93 |  66.32  |   25.70  | 58.82 | 58.82 |
| Ours             | 64.02 |  56.71  |  82.24  | 65.69 | 54.09 | 42.92 |  42.22 | 83.81 | 87.41 | 46.54 |  67.07  |   27.57  | 59.30 | 59.97 |
| Ours + Filtering | 65.66 |  57.56  |  83.55  | 64.81 | 56.41 | 51.08 |  44.18 | 84.62 | 87.04 | 53.78 |  73.43  |   28.61  | 60.15 | 62.37 |
