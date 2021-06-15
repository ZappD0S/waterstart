import asyncio
from abc import ABC, abstractmethod
from collections import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import (
    AsyncContextManager,
    Generic,
    Optional,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")

# TODO: maybe pass to the constructor a series of coroutines (or tasks?)
# that are scheduled and when we close are canceled and awaited?
class Observable(ABC, Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self._setters: list[Callable[[T], None]] = []
        self._call_setters_task: Optional[asyncio.Task[None]] = None

    @abstractmethod
    def _get_async_generator(self) -> AsyncGenerator[T, None]:
        ...

    async def _call_setters(self) -> None:
        gen = self._get_async_generator()
        try:
            async for value in gen:
                for setter in self._setters:
                    setter(value)
        finally:
            await gen.aclose()

    @asynccontextmanager
    async def _add_setter(self, setter: Callable[[T], None]):
        if self._call_setters_task is None:
            self._call_setters_task = asyncio.create_task(self._call_setters())

        self._setters.append(setter)
        try:
            yield
        finally:
            self._setters.remove(setter)
            if self._setters:
                return

            self._call_setters_task.cancel()

            try:
                await self._call_setters_task
            except asyncio.CancelledError:
                pass
            finally:
                self._call_setters_task = None

    # TODO: maybe rename these to `subscribe`?

    def _register_with_iterable(
        self, func: Callable[[T], Optional[U]], it: AsyncIterator[T], maxsize: int = 0
    ) -> AsyncContextManager[AsyncIterator[U]]:
        async def call_setter(setter: Callable[[T], None]) -> None:
            async for value in it:
                setter(value)

        @asynccontextmanager
        async def add_setter(setter: Callable[[T], None]):
            call_setters_task = asyncio.create_task(call_setter(setter))
            try:
                yield
            finally:
                call_setters_task.cancel()

                try:
                    await call_setters_task
                except asyncio.CancelledError:
                    pass

        return self._register(func, add_setter, maxsize)

    def register(
        self, func: Callable[[T], Optional[U]], maxsize: int = 0
    ) -> AsyncContextManager[AsyncIterator[U]]:
        return self._register(func, self._add_setter, maxsize)

    def _register(
        self,
        func: Callable[[T], Optional[U]],
        add_setter: Callable[[Callable[[T], None]], AsyncContextManager[None]],
        maxsize: int,
    ) -> AsyncContextManager[AsyncIterator[U]]:
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize)

        def set_result(val: T):
            if queue.full():
                _ = queue.get_nowait()

            queue.put_nowait(val)

        async def get_generator() -> AsyncGenerator[U, None]:
            while True:
                raw = await queue.get()

                if (val := func(raw)) is not None:
                    yield val

        @asynccontextmanager
        async def get_contextmanager():
            gen = get_generator()

            try:
                async with add_setter(set_result):
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
