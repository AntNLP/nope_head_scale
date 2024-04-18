# Head Scale for NoPE

This repo contains the code for the Head Scale length generalization method.

The paper is currently under ARR 2024 Feb review.

## Models

### NoPE Pretained Model

Coming soon

### Head Scale NoPE Model

Coming soon

## Reproduction

### Environment

Install pytorch and other packages.

```python
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Install Flash-Attention 2 following https://github.com/Dao-AILab/flash-attention

### Data Preparation

For PG19 and proof_pile:

```bash
python data/prepare_lctx.py
```

For TinyLLaMA pretraining data: follow https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md#data-preparation

### Head Scale

To fine-tune the model using head scale proposed in the paper, run the following script

```bash
./script/run_hs.sh
```

### Evaluating perplexity

First prepare PG19 and proof_pile dataset as stated before, then run the following script

```bash
./script/run_ppl.sh
```

Each run would generate a json log file in `logging_dir` with the following format

```json
{
    "model_name": "TinyLlama/tinyLlama-intermediate-checkpoints",
    "logging_dir": "path/to/logging_dir",
    "max_length": 2048,
    "loss": 2.676506847143173,
    "ppl": 14.534234216552017
}
```

Then you can collect and convert multiple runs to markdown table using `fig/tab_ppl.ipynb`

### Passkey Retrieval Task

```bash
./script/run_pass.sh
```

This would generate a csv file in `logging_dir` with the following format and can be visualized using `fig/fig_passkey.ipynb`

```csv
length,depth,position,accuracy
256,0%,9,1.0
256,10%,39,1.0
256,20%,61,0.9
...
4096,60%,2638,0.0
4096,70%,3058,0.0
4096,80%,3484,0.0
4096,90%,3920,0.0
```

### LongBench

Run inference via `pred.py`

```bash
python pred.py
```

And then collect output and evaluate via `eval.py`

```bash
python eval.py
```

For more instructions, please follow [the official repo](https://github.com/THUDM/LongBench)

### Reproducing RoPE baselines

Finetuning RoPE length generation models:

```bash
./script/run_pi.sh
./script/run_yarn.sh
```

Then modify the evaluation script and run. See examples in the script.

```bash
./script/run_ppl.sh
./script/run_pass.sh
```

## Citation

```text
Coming soon
```
