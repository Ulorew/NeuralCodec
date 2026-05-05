def rename_key(d, old, new):
    d[new] = d.pop(old)


def wrap_keys(d, keys, pref="", suff=""):
    for k in keys:
        rename_key(d, k, pref + k + suff)


def freeze_model(model, freeze=True):
    for p in model.parameters():
        p.requires_grad = not freeze
