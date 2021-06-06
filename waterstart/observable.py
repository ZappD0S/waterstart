import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Callable,
    ContextManager,
    Generic,
    Iterator,
    Optional,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")

# TODO: maybe pass to the constructor a series of coroutines (or tasks?)
# that are scheduled and when we close are canceled and awaited?
class Observable(ABC, Generic[T]):
    def __init__(self, maxsize: int = 1) -> None:
        super().__init__()
        self._maxsize = maxsize
        self._setters: list[Callable[[T], None]] = []
        self._call_setters_task: Optional[asyncio.Task[None]] = None

    @abstractmethod
    def _get_async_iterator(self) -> AsyncIterator[T]:
        ...

    async def _call_setters(self) -> None:
        # TODO: do we need to close the generator explicitly?
        async for value in self._get_async_iterator():
            for setter in self._setters:
                setter(value)

    @contextmanager
    def _add_setter(self, setter: Callable[[T], None]) -> Iterator[None]:
        if self._call_setters_task is None:
            self._call_setters_task = asyncio.create_task(self._call_setters())

        self._setters.append(setter)
        try:
            yield
        finally:
            self._setters.remove(setter)
            if not self._setters:
                self._call_setters_task.cancel()
                self._call_setters_task = None

    # TODO: maybe rename these to `subscribe`?

    def _register_with_iterable(
        self, func: Callable[[T], Optional[U]], it: AsyncIterator[T]
    ) -> AsyncContextManager[AsyncIterator[U]]:
        async def call_setter(setter: Callable[[T], None]) -> None:
            async for value in it:
                setter(value)

        @contextmanager
        def add_setter(setter: Callable[[T], None]) -> Iterator[None]:
            call_setters_task = asyncio.create_task(call_setter(setter))
            try:
                yield
            finally:
                call_setters_task.cancel()

        return self._register(func, add_setter)

    def register(
        self, func: Callable[[T], Optional[U]]
    ) -> AsyncContextManager[AsyncIterator[U]]:
        return self._register(func, self._add_setter)

    def _register(
        self,
        func: Callable[[T], Optional[U]],
        add_setter: Callable[[Callable[[T], None]], ContextManager[None]],
    ) -> AsyncContextManager[AsyncIterator[U]]:
        queue: asyncio.Queue[T] = asyncio.Queue(self._maxsize)

        def set_result(val: T):
            if queue.full():
                _ = queue.get_nowait()

            queue.put_nowait(val)

        async def get_generator():
            while True:
                raw = await queue.get()

                if (val := func(raw)) is not None:
                    yield val

        @asynccontextmanager
        async def get_contextmanager():
            gen = get_generator()

            try:
                with add_setter(set_result):
                    yield gen
            finally:
                await gen.aclose()

        return get_contextmanager()

    # async def close(self) -> None:
    #     self._call_setters_task.cancel()
    #     try:
    #         await self._call_setters_task
    #     except asyncio.CancelledError:
    #         pass
