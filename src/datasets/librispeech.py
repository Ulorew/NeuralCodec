import math

import torch
import torchaudio
from torchcodec.decoders import AudioDecoder

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(BaseDataset):
    def __init__(
        self,
        sampling_rate,
        window_size,
        name="train-clean-100",
        fixed_cuts=False,
        *args,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.segment_len = math.floor(window_size * sampling_rate)
        self.fixed_cuts = fixed_cuts

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
            dec = AudioDecoder(fp)

            md = dec.metadata

            sr = md.sample_rate
            duration_s = md.duration_seconds
            duration_d = math.floor(duration_s * sr)

            if sr != self.sampling_rate:
                raise ValueError(
                    f"Inconsistent sampling rate: expected {self.sampling_rate}, found {sr}"
                )

            info = {}

            max_offset = int(duration_d - self.segment_len)

            if max_offset < 0:
                raise ValueError(
                    "Too short audio found. Unsupported."
                )  # TODO: add short audio support

            label = fp.name.rstrip(".flac")
            info.update(
                {
                    "path": str(fp),
                    "label": label,
                    "duration": duration_d,
                    "max_offset": max_offset,
                }
            )

            index.append(info)

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, info):
        if self.fixed_cuts:
            start = info["duration"] - (self.segment_len // 2)
        else:
            start = torch.randint(0, info["max_offset"] + 1, ()).item()

        audio, _ = torchaudio.load(
            info["path"], frame_offset=start, num_frames=self.segment_len
        )
        return audio

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
        data_object = self.load_object(data_dict)
        data_label = data_dict["label"]

        instance_data = {"orig": data_object, "label": data_label}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
