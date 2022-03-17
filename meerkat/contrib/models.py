import torch

from meerkat.contrib.registry import Registry

models = Registry("models")
models.__doc__ = "Registry for pretrained models"

@models.register()
def clip(variant="ViT-B/32", **kwargs):
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    return model, preprocess