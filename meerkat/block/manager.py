from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from typing import Dict, Mapping, Sequence, Union

import numpy as np

from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.blockable import BlockableMixin

from .abstract import BlockIndex
from .ref import BlockRef


class BlockManager(MutableMapping):
    """

    This manager manages all blocks.
    """

    def __init__(self) -> None:
        self._columns: Dict[str, AbstractColumn] = {}
        self._column_to_block_id: Dict[str, int] = {}
        self._block_refs: Dict[int, BlockRef] = {}

    def update(self, block_ref: BlockRef):
        """
        data (): a single blockable object, potentially contains multiple columns
        """
        # can't have the same column living in multiple managers, so we view
        self._columns.update(
            {name: column.view() for name, column in block_ref.items()}
        )

        block_id = id(block_ref.block)
        # check if there already is a block_ref in the manager for this block
        if block_id in self._block_refs:
            self._block_refs[block_id].update(block_ref)
        else:
            self._block_refs[block_id] = block_ref

        self._column_to_block_id.update({name: block_id for name in block_ref.keys()})

    def apply(self, method_name: str = "_get", *args, **kwargs) -> BlockManager:
        """[summary]

        Args:
            fn (str): a function that is applied to a block and column_spec and
                returns a new block and column_spec.
        Returns:
            [type]: [description]
        """
        results = None
        for block_ref in self._block_refs.values():
            result = block_ref.apply(method_name=method_name, *args, **kwargs)
            if results is None:
                results = BlockManager() if isinstance(result, BlockRef) else {}
            results.update(result)

        # apply method to columns not stored in block
        for name, col in self._columns.items():
            if name in results:
                continue
            result = getattr(col, method_name)(*args, **kwargs)
            if results is None:
                results = BlockManager() if isinstance(result, AbstractColumn) else {}

            results[name] = result

        return results

    def consolidate(self):
        block_ref_groups = defaultdict(list)
        for block_ref in self._block_refs.values():
            block_ref_groups[block_ref.block.signature].append(block_ref)

        for block_refs in block_ref_groups.values():
            if len(block_refs) == 0:
                continue

            # remove old block_refs
            for old_ref in block_refs:
                del self._block_refs[id(old_ref.block)]

            # consolidate group
            block_class = block_refs[0].block.__class__
            block_ref = block_class.consolidate(block_refs)
            self.update(block_ref)

    def remove(self, name):
        if name not in self._columns:
            raise ValueError(f"Remove failed: no column '{name}' in BlockManager.")

        del self._columns[name]

        if name in self._column_to_block_id:
            # column is blockable
            block_ref = self._block_refs[self._column_to_block_id[name]]
            del block_ref[name]

            if len(block_ref) == 0:
                del self._block_refs[self._column_to_block_id[name]]

    @property
    def block_indices(self) -> Mapping[str, BlockIndex]:
        return {name: col.index for name, col in self._columns}

    def __getitem__(
        self, index: Union[str, Sequence[str]]
    ) -> Union[AbstractColumn, BlockManager]:
        if isinstance(index, str):
            return self._columns[index]
        elif isinstance(index, Sequence):
            mgr = BlockManager()
            block_id_to_names = defaultdict(list)
            for name in index:
                if name not in self._column_to_block_id:
                    if name in self:
                        # non-blockable column
                        mgr.add_column(col=self._columns[name], name=name)
                    else:
                        raise ValueError(
                            f"BlockManager does not contain column '{name}'."
                        )
                else:
                    # group blockable columns by block
                    block_id_to_names[self._column_to_block_id[name]].append(name)

            # block refs for blockable columns
            for block_id, names in block_id_to_names.items():
                block_ref = self._block_refs[block_id]
                mgr.update(block_ref[names])
            return mgr
        else:
            raise ValueError(
                f"Unsupported index of type {type(index)}passed to `BlockManager`."
            )

    def __setitem__(self, index, data: Union[str, Sequence[str]]):
        if isinstance(data, AbstractColumn):
            self.add_column(data, name=index)
        else:
            raise ValueError(
                f"Can't set item with object of type {type(data)} on BlockManager."
            )

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self._columns)

    def __contains__(self, value):
        return value in self._columns

    def __iter__(self):
        return iter(self._columns)

    def add_column(self, col: AbstractColumn, name: str):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if not col.is_blockable():
            col = col.view()
            self._columns[name] = col

        else:
            # TODO: this should be a view and the _block should get carried
            col = col.view()
            self.update(BlockRef(columns={name: col}, block=col._block))

    @classmethod
    def from_dict(cls, data):
        mgr = cls()
        for name, data in data.items():
            col = AbstractColumn.from_data(data)
            mgr.add_column(col=col, name=name)
        return mgr