import random

import torch
import torchaudio
from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class AudioDataset(BaseDataset):
    def __init__(
        self,
        sampling_rate,
        window_size,
        dataset_name="LibriSpeech",
        name="train-clean-100",
        base_factor=1,
        minimal_length=0.1,
        maximal_length=600,
        fixed_cuts=False,
        custom_index=False,
        shuffle_index=False,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.sampling_rate = sampling_rate
        self.trunc = window_size is not None
        self.base_factor = base_factor
        self.window_size = window_size
        self.minimal_length = minimal_length
        self.maximal_length = maximal_length

        self.fixed_cuts = fixed_cuts
        self.custom_index = custom_index

        index_path = ROOT_PATH / "data" / dataset_name / name / "index.json"

        if index_path.exists() and not custom_index:
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)
        if shuffle_index:
            random.shuffle(index)

        super().__init__(index, *args, **kwargs)

    def trunc_to_factor(self, x):
        return x - (x % self.base_factor)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "data" / self.dataset_name / name
        if not data_path.exists():
            raise ValueError(f"Can't find the dataset at {data_path}")

        for fp in data_path.rglob("*"):
            if fp.suffix.lower() not in {".flac", ".mp3", ".aac"}:
                continue

            decoder = AudioDecoder(str(fp))
            md = decoder.metadata

            sr = md.sample_rate
            duration_d = int(round(md.duration_seconds * sr))
            if not (self.minimal_length <= md.duration_seconds <= self.maximal_length):
                continue

            if sr != self.sampling_rate:
                if sr % self.sampling_rate != 0:
                    raise ValueError(
                        f"Inconsistent sampling rate: expected divisible by {self.sampling_rate}, found {sr}"
                    )

            if (
                self.custom_index
                and self.trunc
                and md.duration_seconds < self.window_size
            ):
                continue

            info = {}

            label = fp.stem
            info.update(
                {
                    "path": str(fp),
                    "label": label,
                    "duration": duration_d,
                    "sampling_rate": sr,
                }
            )

            index.append(info)

        # sort by duration for optimal padding
        index.sort(key=lambda x: x["duration"])

        # write index to disk
        if not self.custom_index:
            write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, info):
        if self.trunc:
            num_frames = int(round(self.window_size * info["sampling_rate"]))
            target_frames = int(round(self.window_size * self.sampling_rate))

            max_offset = int(info["duration"] - num_frames)

            if max_offset < 0:
                start = 0
            elif self.fixed_cuts:
                start = max_offset // 2
            else:
                start = torch.randint(0, max_offset + 1, ()).item()

            audio, sr = torchaudio.load(
                info["path"], frame_offset=start, num_frames=num_frames
            )

            if sr != self.sampling_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=sr,
                    new_freq=self.sampling_rate,
                )

            pad_len = target_frames - audio.shape[-1]
            if pad_len > 0:
                audio = F.pad(audio, (0, pad_len), "replicate")
        else:
            audio, sr = torchaudio.load(info["path"])
            if sr != self.sampling_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=sr,
                    new_freq=self.sampling_rate,
                )

        dur = self.trunc_to_factor(audio.shape[-1])
        audio = audio[..., :dur]
        amp = audio.abs().max()
        # rms = (audio**2).mean().sqrt()
        return audio, amp

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_object, amp = self.load_object(data_dict)
        data_label = data_dict["label"]

        instance_data = {
            "orig": data_object,
            "label": data_label,
            "length": data_object.shape[-1],
            "amp": amp,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data
