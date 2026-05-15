import os
import re
import warnings
from pathlib import Path

import hydra
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from IPython.display import Audio, display

from src.datasets.data_utils import get_dataloaders
from src.metrics.speech_metrics import NISQAMetric, STOIMetric
from src.trainer import GANInferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_SAMPLE_RATE = 16000


def compose_inference_config(overrides=None):
    overrides = list(overrides or [])
    config_dir = ROOT_PATH / "src" / "configs"
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return hydra.compose(config_name="inference", overrides=overrides)


def run_inference(config_overrides=None, save_path=None, device=None):
    os.chdir(ROOT_PATH)
    overrides = list(config_overrides or [])
    if save_path is not None:
        overrides.append(f"inferencer.save_path={save_path}")
    if device is not None:
        overrides.append(f"inferencer.device={device}")

    config = compose_inference_config(overrides)
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = config.inferencer.device

    return run_inference_from_config(config, runtime_device)


def run_inference_from_config(config, infer_device=None):
    if infer_device is None:
        if config.inferencer.device == "auto":
            infer_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            infer_device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, infer_device)
    model = instantiate(config.model).to(infer_device)
    metrics = instantiate(config.metrics)

    output_dir = ROOT_PATH / "outputs" / "saved" / config.inferencer.save_path
    output_dir.mkdir(exist_ok=True, parents=True)

    inferencer = GANInferencer(
        model=model,
        config=config,
        device=infer_device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=output_dir,
        metrics=metrics,
        skip_model_load=False,
    )
    logs = inferencer.run_inference()
    return logs, output_dir


def load_outputs(run_dir, split=None, limit=None, map_loc="cpu"):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    if split is not None:
        search_roots = [run_dir / split]
    else:
        split_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
        search_roots = split_dirs or [run_dir]

    rows = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("output_*.pth"), key=_output_sort_key):
            data = torch.load(path, map_location=map_loc, weights_only=False)
            item = {
                "path": path,
                "split": root.name if root != run_dir else split,
                "output_id": _output_sort_key(path),
                **data,
            }
            item["length"] = int(
                item.get("length", min(item["orig"].shape[-1], item["recon"].shape[-1]))
            )
            rows.append(item)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def outputs_table(outputs):
    rows = []
    for item in outputs:
        rows.append(
            {
                "split": item.get("split"),
                "output_id": item.get("output_id"),
                "seconds": item["length"] / DEFAULT_SAMPLE_RATE,
                "path": str(item["path"]),
            }
        )
    return pl.DataFrame(rows)


def display_audio_pair(item, sample_rate=DEFAULT_SAMPLE_RATE):
    orig, recon = get_audio_pair(item)
    print(f"Original: {item.get('split')} / output_{item.get('output_id')}")
    display(Audio(orig.numpy(), rate=sample_rate))
    print("Reconstruction")
    display(Audio(recon.numpy(), rate=sample_rate))


def plot_waveform_pair(item, sample_rate=DEFAULT_SAMPLE_RATE, title=None):
    orig, recon = get_audio_pair(item)
    residual = recon - orig
    time = np.arange(orig.numel()) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(time, orig.numpy(), linewidth=0.8)
    axes[0].set_title("Original waveform")
    axes[1].plot(time, recon.numpy(), linewidth=0.8)
    axes[1].set_title("Reconstructed waveform")
    axes[2].plot(time, residual.numpy(), linewidth=0.8)
    axes[2].set_title("Residual: reconstruction - original")
    axes[2].set_xlabel("Time, s")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(title or _default_title(item))
    fig.tight_layout()
    return fig


def plot_stft_pair(
    item,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    title=None,
):
    orig, recon = get_audio_pair(item)
    orig_db = _stft_db(orig.numpy(), n_fft=n_fft, hop_length=hop_length)
    recon_db = _stft_db(recon.numpy(), n_fft=n_fft, hop_length=hop_length)
    diff = np.abs(recon_db - orig_db)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    _show_spec(axes[0], orig_db, sample_rate, hop_length, "Original STFT, dB")
    _show_spec(axes[1], recon_db, sample_rate, hop_length, "Reconstructed STFT, dB")
    _show_spec(axes[2], diff, sample_rate, hop_length, "|Difference|, dB")
    fig.suptitle(title or _default_title(item))
    fig.tight_layout()
    return fig


def plot_mel_pair(
    item,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    title=None,
):
    orig, recon = get_audio_pair(item)
    orig_db = _mel_db(orig.numpy(), sample_rate, n_fft, hop_length, n_mels)
    recon_db = _mel_db(recon.numpy(), sample_rate, n_fft, hop_length, n_mels)
    diff = np.abs(recon_db - orig_db)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    _show_spec(axes[0], orig_db, sample_rate, hop_length, "Original mel, dB")
    _show_spec(axes[1], recon_db, sample_rate, hop_length, "Reconstructed mel, dB")
    _show_spec(axes[2], diff, sample_rate, hop_length, "|Difference|, dB")
    fig.suptitle(title or _default_title(item))
    fig.tight_layout()
    return fig


def summarize_dataset_outputs(
    outputs, sample_rate=DEFAULT_SAMPLE_RATE, include_speech_metrics=False
):
    rows = []
    speech_metrics = None

    for item in outputs:
        orig, recon = get_audio_pair(item)
        diff = recon - orig
        mse = torch.mean(diff**2).item()
        signal_power = torch.mean(orig**2).item()
        snr = 10 * np.log10((signal_power + 1e-12) / (mse + 1e-12))
        orig_np = orig.numpy()
        recon_np = recon.numpy()
        row = {
            "split": item.get("split"),
            "output_id": item.get("output_id"),
            "seconds": orig.numel() / sample_rate,
            "mae": torch.mean(torch.abs(diff)).item(),
            "mse": mse,
            "snr_db": snr,
            "orig_rms": float(np.sqrt(np.mean(orig_np**2))),
            "recon_rms": float(np.sqrt(np.mean(recon_np**2))),
            "orig_peak": float(np.max(np.abs(orig_np))),
            "recon_peak": float(np.max(np.abs(recon_np))),
            "stft_l1_db": stft_l1_db(orig_np, recon_np),
            "mel_l1_db": mel_l1_db(orig_np, recon_np, sample_rate),
            "unique_codes": unique_code_count(item.get("codes")),
        }
        if include_speech_metrics:
            if speech_metrics is None:
                speech_metrics = {
                    "stoi": STOIMetric(sample_rate),
                    "nisqa": NISQAMetric(sample_rate),
                }
            row.update(compute_speech_metrics(orig, recon, speech_metrics))
        rows.append(row)

    frame = pl.DataFrame(rows)
    if frame.shape[0] == 0:
        return frame

    numeric = frame.select(cs.numeric())

    return numeric.describe()


def compare_summary_tables(
    named_outputs, sample_rate=DEFAULT_SAMPLE_RATE, include_speech_metrics=False
):
    rows = []

    metrics = [
        "mae",
        "mse",
        "snr_db",
        "stft_l1_db",
        "mel_l1_db",
        "unique_codes",
    ]
    if include_speech_metrics:
        metrics.extend(["stoi", "nisqa"])

    for name, outputs in named_outputs.items():
        summary = summarize_dataset_outputs(
            outputs,
            sample_rate=sample_rate,
            include_speech_metrics=include_speech_metrics,
        )

        if summary.height == 0:
            continue

        row = {"dataset": name}

        mean_row = summary.filter(pl.col("statistic") == "mean")

        if mean_row.height == 0:
            continue

        existing_metrics = [m for m in metrics if m in summary.columns]

        for metric in existing_metrics:
            row[metric] = mean_row.select(metric).item()

        rows.append(row)

    return pl.DataFrame(rows)


def save_summary_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    frame.write_csv(path)
    return path


def compute_speech_metrics(orig, recon, speech_metrics):
    length = torch.tensor([orig.numel()])
    orig_batch = orig.unsqueeze(0)
    recon_batch = recon.unsqueeze(0)
    return {
        "stoi": _metric_value_or_none(
            speech_metrics["stoi"](orig=orig_batch, recon=recon_batch, length=length)
        ),
        "nisqa": _metric_value_or_none(
            speech_metrics["nisqa"](recon=recon_batch, length=length)
        ),
    }


def get_audio_pair(item):
    length = int(
        item.get("length", min(item["orig"].shape[-1], item["recon"].shape[-1]))
    )
    orig = _to_mono(item["orig"])[..., :length].detach().cpu()
    recon = _to_mono(item["recon"])[..., :length].detach().cpu()
    length = min(orig.numel(), recon.numel())
    return orig[:length], recon[:length]


def stft_l1_db(orig, recon, n_fft=1024, hop_length=256):
    return float(
        np.mean(
            np.abs(
                _stft_db(orig, n_fft, hop_length) - _stft_db(recon, n_fft, hop_length)
            )
        )
    )


def mel_l1_db(
    orig, recon, sample_rate=DEFAULT_SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80
):
    orig_db = _mel_db(orig, sample_rate, n_fft, hop_length, n_mels)
    recon_db = _mel_db(recon, sample_rate, n_fft, hop_length, n_mels)
    return float(np.mean(np.abs(orig_db - recon_db)))


def unique_code_count(codes):
    if codes is None:
        return np.nan
    return int(torch.unique(codes.detach().cpu()).numel())


def _metric_value_or_none(value):
    if value is None:
        return None
    return float(value)


def _output_sort_key(path):
    match = re.search(r"output_(\d+)\.pth$", Path(path).name)
    return int(match.group(1)) if match else -1


def _to_mono(audio):
    audio = audio.detach().cpu().float()
    while audio.dim() > 1:
        audio = audio.mean(dim=0)
    return audio


def _stft_db(audio, n_fft=1024, hop_length=256):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)


def _mel_db(audio, sample_rate, n_fft, hop_length, n_mels):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def _show_spec(axis, spec, sample_rate, hop_length, title):
    librosa.display.specshow(
        spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=axis,
    )
    axis.set_title(title)


def _default_title(item):
    return f"{item.get('split', 'sample')} / output_{item.get('output_id', '?')}"


@hydra.main(version_base=None, config_path="../../src/configs", config_name="inference")
def main(config):
    os.chdir(ROOT_PATH)
    set_random_seed(config.inferencer.seed)
    logs, _ = run_inference_from_config(config)
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print("    {:15s}: {}".format(full_key, value))


if __name__ == "__main__":
    main()
