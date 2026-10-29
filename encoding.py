import numpy as np
from manual_features import encode, CAT_NAMES
from feature_library import window_features, encode_auto

DAY_WINDOWS = [("d1", None, 86400), ("d7", None, 604800)]


def encode_manual(h, ids, cat, cut):
    assert h and all(r["seconds"] <= cut for r in h)
    b, c, names, _, flags = encode(h, ids, cat, cut)
    extra, en = window_features(h, ids, cat, cut, windows=DAY_WINDOWS)
    return (
        np.column_stack([b, extra]).astype("float32"),
        c,
        names + [n.replace("auto_", "manual_", 1) for n in en],
        flags,
    )


def encode_library(h, ids, cat, cut, b, names):
    return encode_auto(h, ids, cat, cut, b, names)
