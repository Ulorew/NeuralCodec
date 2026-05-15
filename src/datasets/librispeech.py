import torch
import torchaudio
from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(BaseDataset):
    def __init__(
            self,
            sampling_rate,
            window_size,
            name="train-clean-100",
            base_factor=1,
            fixed_cuts=False,
            custom_index=False,
            *args,
            **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.trunc = window_size is not None
        self.base_factor = base_factor
        self.segment_len = (
            self.trunc_to_factor(int(window_size * sampling_rate))
            if self.trunc
            else None
        )

        self.fixed_cuts = fixed_cuts
        self.custom_index = custom_index

        index_path = ROOT_PATH / "data" / "LibriSpeech" / name / "index.json"

        if index_path.exists() and not custom_index:
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def trunc_to_factor(self, x):
        return x - (x % self.base_factor)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "data" / "LibriSpeech" / name
        if not data_path.exists():
            raise ValueError(f"Can't find the dataset at {data_path}")

        for fp in data_path.rglob("*.flac"):
            decoder = AudioDecoder(str(fp))
            md = decoder.metadata

            sr = md.sample_rate
            duration_d = int(md.duration_seconds * sr)

            if sr != self.sampling_rate:
                raise ValueError(
                    f"Inconsistent sampling rate: expected {self.sampling_rate}, found {sr}"
                )

            if self.custom_index and self.trunc and duration_d < self.segment_len:
                continue

            info = {}

            label = fp.name.rstrip(".flac")
            info.update(
                {
                    "path": str(fp),
                    "label": label,
                    "duration": duration_d,
                }
            )

            index.append(info)

        # write index to disk
        if not self.custom_index:
            write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, info):
        if self.trunc:
            max_offset = int(info["duration"] - self.segment_len)

            if max_offset < 0:
                start = 0
            elif self.fixed_cuts:
                start = max_offset // 2
            else:
                start = torch.randint(0, max_offset + 1, ()).item()

            audio, _ = torchaudio.load(
                info["path"], frame_offset=start, num_frames=self.segment_len
            )

            pad_len = self.segment_len - audio.shape[-1]
            if pad_len > 0:
                audio = F.pad(audio, (0, pad_len), "replicate")
        else:
            dur = self.trunc_to_factor(info["duration"])
            audio, _ = torchaudio.load(info["path"], num_frames=dur)

        # rms = (audio**2).mean().sqrt()
        amp = audio.abs().max()
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

        instance_data = {"orig": data_object, "label": data_label, "amp": amp}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
