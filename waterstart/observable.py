import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Generic,
    MutableSequence,
    Optional,
    TypeVar,
    overload,
)

T = TypeVar("T")
U = TypeVar("U")


class Observabe(ABC, Generic[T]):
    def __init__(self, maxsize: int = 1) -> None:
        super().__init__()
        self._maxsize = maxsize
        self._setters: MutableSequence[Callable[[T], None]] = []
        self._dispatch_task = asyncio.create_task(self._dispatch_messages())

    @abstractmethod
    def _get_async_iterator(self) -> AsyncIterator[T]:
        ...

    async def _dispatch_messages(self) -> None:
        # TODO: do we need to close the generator explicitly?
        async for value in self._get_async_iterator():
            for setter in self._setters:
                setter(value)

    @overload
    def register(self) -> AsyncContextManager[AsyncIterator[T]]:
        ...

    @overload
    def register(
        self, func: Callable[[T], Optional[U]]
    ) -> AsyncContextManager[AsyncIterator[U]]:
        ...

    def register(self, func=None):
        queue = asyncio.Queue(self._maxsize)

        if func is None:
            func = lambda x: x

        def set_result(val):
            if queue.full():
                _ = queue.get_nowait()

            queue.put_nowait(val)

        async def get_generator():
            while True:
                raw = await queue.get()

                val = func(raw)

                if val is not None:
                    yield val

        @asynccontextmanager
        async def _get_contextmanager() -> AsyncIterator:
            self._setters.append(set_result)
            try:
                yield gen
            finally:
                await gen.aclose()
                self._setters.remove(set_result)

        gen = get_generator()
        return _get_contextmanager()

    async def close(self) -> None:
        self._dispatch_task.cancel()
        try:
            await self._dispatch_task
        except asyncio.CancelledError:
            pass
