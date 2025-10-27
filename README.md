# Jailbreak Transferability Analysis

Code for replicating results in [Jailbreak Transferability Emerges from Shared Representations]()

### Setup

Install necessary packages
```bash
pip install git+https://github.com/dsbowen/strong_reject.git@main
pip install git+https://github.com/andyzoujm/representation-engineering.git@main
pip install vllm seaborn scikit-learn matplotlib numpy scipy
```

Set environment variables
```bash
export HF_TOKEN=[...]
export OPENAI_API_KEY=[...]
```

### Downloading the data

Our training and results data can be downloaded [here]().

### Sampling and judging generations

Run the following command to sample and judge generations to all StrongREJECT jailbreaks for `meta-llama/Llama-3.1-8B-Instruct`:
```bash
python -m src.generate_jailbreak_responses --target-model llama3.1-8b
```
Throughout this repository we use model shortnames defined in `src/config.py`. To run on new models, just add the model shortname here.

### Computing model similarities

To compute the model similarities, we first generate k-nearest neighbor graphs on `tatsu-lab/alpaca` for all models. Run the following command for all target models of interest:
```bash
python -m src.generate_knn_graphs --target-model llama3.1-8b
```
Compute all pairwise model similarities by running the jupyter notebook `notebooks/compute_model_similarities.ipynb`. This notebook computes all pairwise model similarities and saves them to `notebooks/pairwise_knn_layer_dfs.pkl`.

### Observational analysis

Our observational results can be replicated using the jupyter notebook `notebooks/analyze_model_transfer.ipynb`.

### Distillation

We provide code for performing distillation in the `distillations/` subdirectory. The distillation code is adapted from [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
