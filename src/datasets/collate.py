import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    origs = [elem["orig"].transpose(0, 1) for elem in dataset_items]
    origs = pad_sequence(origs, batch_first=True).transpose(1, 2)
    lengths = [elem.get("length", elem["orig"].shape[-1]) for elem in dataset_items]

    return {
        "orig": origs,
        "length": torch.tensor(lengths),
        "amp": torch.stack([elem["amp"] for elem in dataset_items]),
        "label": [elem["label"] for elem in dataset_items],
    }
