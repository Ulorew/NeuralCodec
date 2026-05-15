# SoundStream Neural Codec Replica

This repository contains a replica of [SoundStream audio codec](https://arxiv.org/abs/2107.03312),
including project-style training, inference, demo, and analysis code for 16 kHz speech.
The model uses an encoder-decoder with residual vector quantization, adversarial training, and the
16 kHz stride schedule `[2, 4, 5, 5]`.

## Demo

The required demo notebook is [src/notebooks/demo.ipynb](src/notebooks/demo.ipynb).
In a fresh Colab session, set `AUDIO_SOURCE` to an audio URL and run all cells. The notebook clones
the repository, installs dependencies, downloads `demo_data/model.pth`, runs the codec, and displays
the original and re-synthesized audio.

The same resources can be downloaded locally with:

```bash
python prepare_demo.py
```

## Setup

For a basic CPU setup, run:

```bash
pip install -r requirements.txt
```

For GPU setup, install `torch` and `torchaudio` according to your system configuration.
Follow [the official instructions](https://pytorch.org/get-started/locally/).
Please note that the default installation command may not include `torchaudio`; add it explicitly.
For CUDA 12.8, use:

```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining packages:

```bash
pip install -r requirements_common.txt
```

## Data

The training setup expects the Kaggle LibriSpeech layout under `data/LibriSpeech`:

```text
data/LibriSpeech/train-clean-100
data/LibriSpeech/test-clean
```

Training uses `train-clean-100`. Final evaluation and the analysis notebook use full utterances from
`test-clean`. Dataset configs live in [src/configs/datasets](src/configs/datasets).

## Inference

For single-file inference, use the [demo notebook](src/notebooks/demo.ipynb). For dataset inference:

```bash
python prepare_demo.py
python inference.py inferencer.from_pretrained=demo_data/model.pth datasets=librispeech_full inferencer.save_path=librispeech_full
```

The inferencer writes reconstructed audio and metric tables to the configured output directory.

## Training

The default training config is [src/configs/baseline.yaml](src/configs/baseline.yaml).
It trains on 0.5 second random crops of 16 kHz `train-clean-100` speech and logs individual loss terms,
audio examples, metrics, and codebook perplexity through Comet ML by default.

```bash
python train.py
```

Configure Comet credentials before training if they are not already available in your environment.

## Analysis and Results

The report analysis notebook is [src/notebooks/analysis.ipynb](src/notebooks/analysis.ipynb).
It contains qualitative waveform/spectrogram/audio comparisons for LibriSpeech, external English speech,
and Russian speech, plus quantitative statistics and final full-test metrics.

Current full LibriSpeech `test-clean` metrics from the analysis notebook:

| Dataset | STOI | NISQA |
| --- | ---: | ---: |
| LibriSpeech test-clean full | 0.8516 | 3.4311 |

## Details

For full dataset inference, we keep files in the default length range of 0.1 to 600 seconds. For the compact
analysis subsets, we use 2 to 40 second files to avoid excessive padding and short metric artifacts.
Additionally, we skip metric calculation for samples that are too short for NISQA or STOI.

We use adversarial loss linear warmup. It starts with zero weight and gains full weight in the middle of training.

We don't implement RVQ k-means initialization, leaving it on dead code replacement.

Training lasts for 200 epochs, each one consists of 100 steps. Batch size is 64.

For additional configuration details, check [src/configs/baseline.yaml](src/configs/baseline.yaml).



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
