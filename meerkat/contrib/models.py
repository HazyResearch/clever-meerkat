import torch

from meerkat.contrib.registry import Registry
from functools import partial

models = Registry("models")
models.__doc__ = "Registry for pretrained models"

apis = Registry("apis")
apis.__doc__ = "Registry for model APIs"

@models.register()
def clip(variant="ViT-B/32", **kwargs):
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    return model, preprocess

@apis.register()
def codex(secret, engine='code-davinci-001', **kwargs):
    import openai

    openai.api_key = secret

    return partial(
        openai.Completion.create,
        engine=engine,
        **kwargs,
    )
