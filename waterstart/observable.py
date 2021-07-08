import asyncio
from abc import ABC, abstractmethod
from collections import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    AsyncContextManager,
    Collection,
    Generic,
    Optional,
    Set,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class State(Generic[T]):
    setters: Set[Callable[[T], None]]
    gen: AsyncGenerator[T, None]
    call_setters_task: asyncio.Task[None]


# TODO: maybe pass to the constructor a series of coroutines (or tasks?)
# that are scheduled and when we close are canceled and awaited?
class Observable(ABC, Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[State[T]] = None

    @abstractmethod
    def _get_async_generator(self) -> AsyncGenerator[T, None]:
        ...

    @staticmethod
    async def _call_setters(
        gen: AsyncIterator[T], setters: Collection[Callable[[T], None]]
    ) -> None:
        async for value in gen:
            for setter in setters:
                setter(value)

    @asynccontextmanager
    async def _add_setter(
        self, setter: Callable[[T], None]
    ) -> AsyncGenerator[None, None]:
        if (state := self._state) is None:
            setters: Set[Callable[[T], None]] = set()
            gen = self._get_async_generator()
            call_setters_task = asyncio.create_task(self._call_setters(gen, setters))
            state = self._state = State(setters, gen, call_setters_task)

        setters = state.setters
        setters.add(setter)
        try:
            yield
        finally:
            setters.remove(setter)
            if not setters:

                call_setters_task = state.call_setters_task
                call_setters_task.cancel()

                try:
                    await call_setters_task
                except asyncio.CancelledError:
                    pass
                finally:
                    await state.gen.aclose()
                    self._state = None

    # TODO: maybe rename these to `subscribe`?

    def _register_with_iterable(
        self, func: Callable[[T], Optional[U]], it: AsyncIterator[T], maxsize: int = 0
    ) -> AsyncContextManager[AsyncIterator[U]]:
        @asynccontextmanager
        async def add_setter(setter: Callable[[T], None]):
            call_setters_task = asyncio.create_task(self._call_setters(it, {setter}))
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

    @asynccontextmanager
    async def _register(
        self,
        func: Callable[[T], Optional[U]],
        add_setter: Callable[[Callable[[T], None]], AsyncContextManager[None]],
        maxsize: int,
    ) -> AsyncIterator[AsyncIterator[U]]:
        def set_result(val: T):
            if queue.full():
                _ = queue.get_nowait()

            queue.put_nowait(val)

        async def get_generator() -> AsyncGenerator[U, None]:
            while True:
                raw = await queue.get()

                if (val := func(raw)) is not None:
                    yield val

        gen = get_generator()
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize)

        try:
            async with add_setter(set_result):
                yield gen
        finally:
            await gen.aclose()

    # async def close(self) -> None:
    #     self._call_setters_task.cancel()
    #     try:
    #         await self._call_setters_task
    #     except asyncio.CancelledError:
    #         pass
