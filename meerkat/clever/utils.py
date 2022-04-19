
import meerkat as mk

class DeferredOp:

    def __init__(self, op: callable, out_col: str, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.out_col = out_col

    def __call__(self, data: mk.DataPanel, **kwargs):
        return self.op(data=data, *self.args, **self.kwargs, **kwargs)

