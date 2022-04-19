import os
from typing import Callable, Union

import meerkat as mk
import torch

from .encoder import Encoder

from meerkat.contrib.registry import Registry
from .bit import bit
from .clip import clip
from ..utils import DeferredOp

__all__ = ["clip", "bit"]

encoders = Registry(name="encoders")

encoders.register(clip, aliases=[])
encoders.register(bit, aliases=[])


def infer_modality(col: mk.AbstractColumn):

    if isinstance(col, mk.ImageColumn):
        return "image"
    elif isinstance(col, mk.PandasSeriesColumn):
        return "text"
    else:
        raise ValueError(f"Cannot infer modality from colummn of type {type(col)}.")


def embed(
    input_col: str,
    data: mk.DataPanel = None,
    encoder: Union[str, Encoder] = "clip",
    modality: str = None,
    out_col: str = None,
    device: Union[int, str] = "cpu",
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
    overwrite: bool = False,
    **kwargs,
) -> mk.DataPanel:
    """Embed a column of data with an encoder from the encoder registry.

    Examples
    --------
    Suppose you have an Image dataset (e.g. Imagenette, CIFAR-10) loaded into a
    `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_. You can embed the
    images in the dataset with CLIP using a code snippet like:

    .. code-block:: python

        import meerkat as mk
        from domino import embed

        dp = mk.datasets.get("imagenette")

        dp = embed(
            data=dp,
            input_col="img",
            encoder="clip"
        )


    Args:
        data (mk.DataPanel): A DataPanel containing the data to embed.
        input_col (str): The name of the column to embed.
        encoder (Union[str, Encoder], optional): Name of the encoder to use. List
            supported encoders with ``domino.encoders``. Defaults to "clip".
            Alternatively, pass an :class:`~domino._embed.encoder.Encoder` object
            containing a custom encoder.
        modality (str, optional): The modality of the data to be embedded. Defaults to
            None, in which case the modality is inferred from the type of the input
            column.
        out_col (str, optional): The name of the column where the embeddings are stored.
            Defaults to None, in which case it is ``"{encoder}({input_col})"``.
        device (Union[int, str], optional): The device on which. Defaults to "cpu".
        mmap_dir (str, optional): The path to directory where a memory-mapped file
            containing the embeddings will be written. Defaults to None, in which case
            the embeddings are not memmapped.
        num_workers (int, optional): Number of worker processes used to load the data
            from disk. Defaults to 4.
        batch_size (int, optional): Size of the batches to  used . Defaults to 128.
        **kwargs: Additional keyword arguments are passed to the encoder. To see
            supported arguments for each encoder, see the encoder documentation (e.g.
            :func:`~domino._embed.clip`).

    Returns:
        mk.DataPanel: ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """

    if out_col is None:
        # important that this is done before locals below, so the correct out_col is
        # passed
        out_col = f"{encoder}({input_col})"

    if data is None:
        kwargs = locals()
        kwargs.pop("data")
        return DeferredOp(op=embed, **kwargs)            

    if modality is None:

        modality = infer_modality(col=data[input_col])

    
    if (out_col in data) and not overwrite:
        return data 

    encoder = encoders.get(encoder, device=device, **kwargs)

    if modality not in encoder:
        raise ValueError(f'Encoder "{encoder}" does not support modality "{modality}".')

    encoder = encoder[modality]

    return _embed(
        data=data,
        input_col=input_col,
        out_col=out_col,
        encode=encoder.encode,
        preprocess=encoder.preprocess,
        collate=encoder.collate,
        device=device,
        mmap_dir=mmap_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def _embed(
    data: mk.DataPanel,
    input_col: str,
    out_col: str,
    encode: Callable,
    preprocess: Callable,
    collate: Callable,
    device: int = None,
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if preprocess is not None:
        embed_input = data[input_col].to_lambda(preprocess)
    else:
        embed_input = data[input_col]

    if collate is not None:
        embed_input.collate_fn = collate
    
    with torch.no_grad():
        data[out_col] = embed_input.map(
            lambda x: encode(x.data.to(device)).cpu().detach().numpy(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap_dir is not None,
            mmap_path=None
            if mmap_dir is None
            else os.path.join(mmap_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return data
