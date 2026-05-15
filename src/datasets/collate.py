import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    origs = [elem["orig"].transpose(0, 1) for elem in dataset_items]
    origs = pad_sequence(origs, batch_first=True).transpose(1, 2)

    return {
        "orig": origs,
        "length" : torch.tensor([elem["length"] for elem in dataset_items]),
        "amp": torch.stack([elem["amp"] for elem in dataset_items]),
        "label": [elem["label"] for elem in dataset_items],
    }
