import math

import numpy as np
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ExampleDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, sampling_rate, window_size, name="train-clean-100", *args, **kwargs
    ):
        self.sampling_rate = sampling_rate
        self.segment_len = math.floor(window_size * sampling_rate)

        index_path = ROOT_PATH / "data" / "LibriSpeech" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "data" / "LibriSpeech" / name
        if not data_path.exists():
            raise ValueError(f"Can't find the dataset at {data_path}")

        for fp in data_path.rglob("*.flac"):
            label = fp.name.rstrip(".flac")
            index.append({"path": str(fp), "label": label})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, path):
        """
        Load audio from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """

        dec = AudioDecoder(path)
        md = dec.metadata

        sr = md.sample_rate
        duration_s = md.duration_seconds

        if sr != self.sampling_rate:
            raise ValueError(
                f"Inconsistent sampling rate: expected {self.sampling_rate}, found {sr}"
            )

        max_offset = int(math.floor(duration_s * sr) - self.segment_len)

        if max_offset < 0:
            raise ValueError(
                "Too short audio found. Unsupported."
            )  # TODO: add short audio support

        start = torch.randint(0, max_offset + 1, ()).item()

        audio, _ = torchaudio.load(
            path, frame_offset=start, num_frames=self.segment_len
        )
        return audio
