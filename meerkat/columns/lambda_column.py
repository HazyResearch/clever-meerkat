from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Sequence, Union

import numpy as np
import yaml
from libcst import Lambda

import meerkat as mk
from meerkat.block.abstract import BlockView
from meerkat.block.lambda_block import LambdaBlock, LambdaCellOp, LambdaOp
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import AbstractColumn
from meerkat.datapanel import DataPanel
from meerkat.display import lambda_cell_formatter
from meerkat.errors import ConcatWarning
from meerkat.tools.lazy_loader import LazyLoader

Image = LazyLoader("PIL.Image")


logger = logging.getLogger(__name__)


class LambdaCell(AbstractCell):
    def __init__(self, data: LambdaCellOp, output_key: Union[None, str, int] = None):
        self._data = data

    @property
    def data(self) -> object:
        """Get the data associated with this cell."""
        return self._data

    def get(self, *args, **kwargs):
        return self.data.get()

    def __eq__(self, other):
        return (other.__class__ == self.__class__) and (self.data == other.data)

    def __repr__(self):
        name = getattr(self.data.fn, "__qualname__", repr(self.data.fn))
        return f"LambdaCell(fn={name})"


class LambdaColumn(AbstractColumn):

    block_class: type = LambdaBlock

    def __init__(
        self,
        data: Union[LambdaOp, BlockView],
        output_type: type = None,
        *args,
        **kwargs,
    ):
        super(LambdaColumn, self).__init__(data, *args, **kwargs)

        self._output_type = output_type

    def _set(self, index, value):
        raise ValueError("Cannot setitem on a `LambdaColumn`.")

    def fn(self, data: object):
        """Subclasses like `ImageColumn` should be able to implement their own
        version."""
        raise NotImplementedError

    def _create_cell(self, data: object) -> LambdaCell:
        return LambdaCell(fn=self.fn, data=data)

    def _get_cell(self, index: int, materialize: bool = True):
        if materialize:
            return self.fn(self._data._get(index, materialize=True))
        else:
            return self._create_cell(data=self._data._get(index, materialize=False))

    def _get_batch(self, indices: np.ndarray, materialize: bool = True):
        if materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            data = self.collate(
                [self._get_cell(int(i), materialize=True) for i in indices]
            )
            if self._output_type is not None:
                data = self._output_type(data)
            return data
        else:
            return self._data.lz[indices]

    def _get(self, index, materialize: bool = True, _data: np.ndarray = None):
        index = self._translate_index(index)
        data = self.data._get(index=index, materialize=materialize)
        if isinstance(index, int):
            if materialize:
                return data
            else:
                return LambdaCell(data=data)

        elif isinstance(index, np.ndarray):
            # support for blocks
            if materialize:
                # materialize could change the data in unknown ways, cannot clone
                return self.__class__.from_data(data=self.collate(data))
            else:
                return self._clone(data=data)

    @classmethod
    def _state_keys(cls) -> Collection:
        return super()._state_keys() | {"fn", "_output_type"}

    @staticmethod
    def concat(columns: Sequence[LambdaColumn]):
        for c in columns:
            if c.fn != columns[0].fn:
                warnings.warn(
                    ConcatWarning("Concatenating LambdaColumns with different `fn`.")
                )
                break

        return columns[0]._clone(mk.concat([c._data for c in columns]))

    def _write_data(self, path):
        # TODO (Sabri): avoid redundant writes in dataframes
        return self.data.write(os.path.join(path, "data"))

    def is_equal(self, other: AbstractColumn) -> bool:
        if other.__class__ != self.__class__:
            return False
        if self.fn != other.fn:
            return False

        return self.data.is_equal(other.data)

    @staticmethod
    def _read_data(path: str):
        meta = yaml.load(
            open(os.path.join(path, "data", "meta.yaml")),
            Loader=yaml.FullLoader,
        )
        if issubclass(meta["dtype"], AbstractColumn):
            return AbstractColumn.read(os.path.join(path, "data"))
        else:
            return DataPanel.read(os.path.join(path, "data"))

    @staticmethod
    def _get_default_formatter() -> Callable:
        return lambda_cell_formatter

    def _repr_cell(self, idx):
        return self.lz[idx]
