# SoundStream Neural Codec replica

This repository contains a replica of [SoundStream audio codec](https://arxiv.org/abs/2107.03312),
including training and inference code. The modifications mostly consist of training techniques and model design choices
the original authors didn't explicitly mention in the article.

## Demo

For a quick setup & inference guide, check [demo notebook](src/notebooks/demo.ipynb).

You can download demo model and audio, using the [download script](prepare_demo.py), or more explicitly, using

```python
from src.utils.download import resolve_input

model_path = resolve_input(model_source, DATA_DIR)
audio_path = resolve_input(audio_source, DATA_DIR)
```

with arbitrary sources, including your local files.


## Setup

For a basic `cpu` setup, run

```bash
pip install -r requirements.txt
```

For `gpu` setup, you have to manually install `torch & torchaudio` according to your system configuration.
Follow [the official instructions](https://pytorch.org/get-started/locally/).
Please note that the default installation command doesn't include `torchaudio` and you have to add it. Additionally, the
project
uses `torch 2.11.0` release. For example,
if you have `cuda 12.8`, you may use

```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Then you can install the remaining packages using
```bash
pip install -r requirements_common.txt
```

## Adding the dataset 

## Inference

## Training

## Details

We limit the data used for evaluation with 60 seconds.

## Citation

This project is a replica inspired by SoundStream. If you use this code or model design, please cite the original
SoundStream paper:

  ```bibtex
@misc{zeghidour2021soundstreamendtoendneuralaudio,
      title={SoundStream: An End-to-End Neural Audio Codec}, 
      author={Neil Zeghidour and Alejandro Luebs and Ahmed Omran and Jan Skoglund and Marco Tagliasacchi},
      year={2021},
      eprint={2107.03312},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2107.03312}, 
}