from __future__ import annotations

import time
from typing import Any, Callable, Optional, Protocol, Tuple

import numpy as np


Inputs = Tuple[np.ndarray, ...]
Outputs = Tuple[Any, ...]
Meta = dict
SaveHook = Callable[..., None]


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
    __slots__ = ("_prepare", "_backend", "_infer", "_process", "_save_hook", "class_names")

    def __init__(
        self,
        prepare: PrepareFn,
        backend: BackendAdapter,
        process: PostprocessFn,
        *,
        save_hook: Optional[SaveHook] = None,
    ) -> None:
        self._prepare = prepare
        self._backend = backend
        self._infer = backend.infer
        self._process = process
        self._save_hook = save_hook
        self.class_names = getattr(backend, "class_names", None)

    def __call__(self, raw_input: Any, **process_kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        verbose = process_kwargs.get("verbose", False)
        
        # Preprocessing
        t0 = time.perf_counter()
        inputs, meta = self._prepare(raw_input)
        t1 = time.perf_counter()
        preprocess_time = (t1 - t0) * 1000  # Convert to ms
        
        # Inference
        t2 = time.perf_counter()
        if process_kwargs:
            outputs = self._infer(inputs, **process_kwargs)
        else:
            outputs = self._infer(inputs)
        t3 = time.perf_counter()
        inference_time = (t3 - t2) * 1000  # Convert to ms
        
        # Postprocessing
        t4 = time.perf_counter()
        if process_kwargs:
            result = self._process(outputs, meta, **process_kwargs)
        else:
            result = self._process(outputs, meta)
        t5 = time.perf_counter()
        postprocess_time = (t5 - t4) * 1000  # Convert to ms
        
        # Log timing if verbose
        if verbose:
            total_time = preprocess_time + inference_time + postprocess_time
            print(
                f"Pipeline timing: preprocess={preprocess_time:.2f}ms, "
                f"inference={inference_time:.2f}ms, "
                f"postprocess={postprocess_time:.2f}ms, "
                f"total={total_time:.2f}ms"
            )

        if self._save_hook is not None and process_kwargs.get("save"):
            try:
                self._save_hook(
                    result[0],
                    result[1],
                    result[2],
                    meta,
                    project=process_kwargs.get("project"),
                    name=process_kwargs.get("name"),
                    class_names=self.class_names,
                    verbose=verbose,
                )
            except Exception as exc:
                if verbose:
                    print(f"Pipeline save hook failed: {exc}")

        return result

    predict = __call__

    def close(self) -> None:
        close_fn: Optional[Callable[[], None]] = getattr(self._backend, "close", None)
        if close_fn is not None:
            close_fn()
