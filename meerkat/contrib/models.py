from meerkat.contrib.registry import Registry

models = Registry("models")
models.__doc__ = "Registry for pretrained models"

@models.register()
def clip()