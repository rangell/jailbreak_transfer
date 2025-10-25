# Jailbreak Transferability Analysis

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

### Computing generations and judging

```bash
python -m src.generate_jailbreak_responses --target-model llama3.1-8b --output-dir results/strong_reject_responses/
```

### Computing model similarities

```bash
python -m src.generate_knn_graphs --target-model llama3.1-8b
```

### Analysis

Pointer to notebooks


### Distillation

Pointer to distillation code
