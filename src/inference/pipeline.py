from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Tuple

import numpy as np


Inputs = Tuple[np.ndarray, ...]
Outputs = Tuple[Any, ...]
Meta = dict


class BackendAdapter(Protocol):
    def infer(self, inputs: Inputs, **kwargs: Any) -> Outputs:  # pragma: no cover - protocol definition
        ...

    def close(self) -> None:  # pragma: no cover - optional
        ...


PrepareFn = Callable[[Any], Tuple[Inputs, Meta]]


class PostprocessFn(Protocol):
    def __call__(
        self,
        outputs: Outputs,
        meta: Meta,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - protocol definition
        ...


class ModelPipeline:
    __slots__ = ("_prepare", "_backend", "_infer", "_process", "class_names")

    def __init__(
        self,
        prepare: PrepareFn,
        backend: BackendAdapter,
        process: PostprocessFn,
    ) -> None:
        self._prepare = prepare
        self._backend = backend
        self._infer = backend.infer
        self._process = process
        self.class_names = getattr(backend, "class_names", None)

    def __call__(self, raw_input: Any, **process_kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs, meta = self._prepare(raw_input)
        if process_kwargs:
            outputs = self._infer(inputs, **process_kwargs)
            return self._process(outputs, meta, **process_kwargs)

        outputs = self._infer(inputs)
        return self._process(outputs, meta)

    predict = __call__

    def close(self) -> None:
        close_fn: Optional[Callable[[], None]] = getattr(self._backend, "close", None)
        if close_fn is not None:
            close_fn()

