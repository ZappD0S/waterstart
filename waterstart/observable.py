import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Collection, Generic, Optional, Set, TypeVar, Union


T = TypeVar("T")
U = TypeVar("U")


class Observable(ABC, Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self._setters: Set[Callable[[Union[T, Exception]], None]] = set()
        self._call_setters_task = asyncio.create_task(
            self._call_setters(self._get_async_generator(), self._setters),
            name="call_setters",
        )

    @abstractmethod
    def _get_async_generator(self) -> AsyncGenerator[T, None]:
        ...

    @staticmethod
    async def _call_setters(
        gen: AsyncIterable[T],
        setters: Collection[Callable[[Union[T, Exception]], None]],
    ) -> None:
        try:
            async for value in gen:
                for setter in setters:
                    setter(value)
        except Exception as e:
            for setter in setters:
                setter(e)

            raise

    @asynccontextmanager
    async def _add_setter(
        self,
        setter: Callable[[Union[T, Exception]], None],
    ) -> AsyncIterator[None]:
        setters = self._setters
        assert setter not in setters

        print(f"about to add setter, n: {len(setters)}")
        setters.add(setter)
        print(f"added setter, n: {len(setters)}")

        try:
            yield
        finally:
            print(f"about to remove setter, n: {len(setters)}")
            setters.remove(setter)
            print(f"removed setter, n: {len(setters)}")

    # TODO: maybe rename this to `subscribe`?
    @asynccontextmanager
    async def register(
        self,
        func: Callable[[T], Optional[U]],
        maxsize: int,
    ) -> AsyncIterator[AsyncIterator[U]]:
        # TODO: we might make this async and use put with a timeout (wait_for)
        # if it fails we use get_nowait and put_nowait
        def set_result(val: Union[T, Exception]):
            if queue.full():
                _ = queue.get_nowait()

            queue.put_nowait(val)

        async def get_generator() -> AsyncGenerator[U, None]:
            while True:
                raw = await queue.get()

                if isinstance(raw, Exception):
                    raise raw

                if (val := func(raw)) is not None:
                    yield val

        gen = get_generator()
        queue: asyncio.Queue[Union[T, Exception]] = asyncio.Queue(maxsize)

        try:
            async with self._add_setter(set_result):
                yield gen
        finally:
            await gen.aclose()

    async def aclose(self) -> None:
        assert not self._setters

        self._call_setters_task.cancel()

        try:
            await self._call_setters_task
        except asyncio.CancelledError:
            pass
