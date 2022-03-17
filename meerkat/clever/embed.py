from meerkat.contrib.models import models as MODEL_REGISTRY
import meerkat as mk
import torch
from torch import nn
import os
from typing import Callable


def embed(
    dp: mk.DataPanel,
    input_col: str,
    model_name: str,
    out_col: str = None,
    model_dir: str = None,
    device: int = None,
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
    **kwargs
):
    """Embed a model from the model registry.

    Args:
        dp (mk.DataPanel): DataPanel to embed.
        model_name (str): Name of the model to embed.
        model_dir (str, optional): Directory to look for the model. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    model, preprocess = MODEL_REGISTRY.get(model_name, model_dir=model_dir, **kwargs)

    def _embed_batch(batch: torch.Tensor):
        return model.encode_image(batch)

    _embed(
        dp=dp,
        input_col=input_col,
        out_col=out_col,
        embedder=_embed_batch,
        preprocesser=preprocess,
        device=device,
        mmap_dir=mmap_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def _embed(
    dp: mk.DataPanel,
    input_col: str,
    embedder: Callable,
    preprocesser: Callable = None,
    out_col: str = None,
    device: int = None,
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if preprocesser is not None:
        embed_input = dp[input_col].to_lambda(preprocesser)
    else:
        embed_input = dp[input_col]

    with torch.no_grad():
        dp["emb"] = embed_input.map(
            lambda x: embedder(x.data.to(device)).cpu().detach().numpy(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap_dir is None,
            mmap_path=None
            if mmap_dir is None
            else os.path.join(mmap_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return dp
