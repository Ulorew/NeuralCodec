def rename_key(d, old, new):
    d[new] = d.pop(old)


def add_key_suffix(d, keys, suff):
    for k in keys:
        rename_key(d, k, k + suff)


def freeze_model(model, freeze=True):
    for p in model.parameters():
        p.requires_grad = not freeze
