import torch


def collate_fn(dataset_items: list[dict]):
    return {
        "orig": torch.stack([elem["orig"] for elem in dataset_items]),
        "label": [elem["label"] for elem in dataset_items],
    }
