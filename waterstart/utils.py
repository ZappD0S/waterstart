from __future__ import annotations
import asyncio

from asyncio import Task
from collections.abc import Sequence, AsyncIterator, AsyncGenerator, Awaitable
from types import TracebackType
from typing import Generic, Optional, TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    from _typeshed import SupportsLessThan


def is_sorted(seq: Sequence[SupportsLessThan]) -> bool:
    return all(a < b for a, b in zip(seq[:-1], seq[1:]))


def is_contiguous(seq: Sequence[int]) -> bool:
    return all(b - a == 1 for a, b in zip(seq[:-1], seq[1:]))


T = TypeVar("T")
U = TypeVar("U")


class ComposableAsyncIterator(Generic[T], AsyncGenerator[T, None]):
    def __init__(self, task_to_it: dict[Task[T], AsyncIterator[T]]) -> None:
        self._task_to_it = task_to_it

    @staticmethod
    def from_it(it: AsyncIterator[U]) -> ComposableAsyncIterator[U]:
        if isinstance(it, ComposableAsyncIterator):
            return ComposableAsyncIterator(it._task_to_it)  # type: ignore

        task_to_it = {asyncio.create_task(it.__anext__()): it}
        return ComposableAsyncIterator(task_to_it)

    def __or__(self, other: AsyncIterator[U]) -> ComposableAsyncIterator[Union[T, U]]:
        task_to_it: dict[Task[Union[T, U]], AsyncIterator[Union[T, U]]]
        task_to_it = self._task_to_it.copy()  # type: ignore

        task_to_it[asyncio.create_task(other.__anext__())] = other  # type: ignore
        return ComposableAsyncIterator(task_to_it)

    def __aiter__(self) -> AsyncGenerator[T, None]:
        return self

    async def __anext__(self) -> T:
        task_to_it = self._task_to_it

        while True:
            done_tasks, _ = await asyncio.wait(
                task_to_it, return_when=asyncio.FIRST_COMPLETED
            )

            if not done_tasks:
                raise StopAsyncIteration()

            for task in done_tasks:
                it = task_to_it.pop(task)

                try:
                    val = await task
                except StopAsyncIteration:
                    continue

                task_to_it[asyncio.create_task(it.__anext__())] = it
                return val

    def asend(self, value: None) -> Awaitable[T]:
        return super().asend(value)

    def athrow(
        self,
        typ: BaseException,
        val: None,
        tb: Optional[TracebackType],
    ) -> Awaitable[T]:
        return super().athrow(typ, val, tb)

    async def aclose(self) -> None:
        for task in self._task_to_it:
            task.cancel()

            try:
                await task
            except Exception:
                pass

        self._task_to_it.clear()


# async def combine_iterators(
#     it1: AsyncIterator[T], it2: AsyncIterator[U]
# ) -> AsyncIterator[Union[T, U]]:

#     task_to_it: dict[Task[Union[T, U]], AsyncIterator[Union[T, U]]]
#     task_to_it = {  # type: ignore
#         asyncio.create_task(it1.__anext__()): it1,
#         asyncio.create_task(it2.__anext__()): it2,
#     }

#     while task_to_it:
#         dones, _ = await asyncio.wait(task_to_it, return_when=asyncio.FIRST_COMPLETED)

#         for done in dones:
#             it = task_to_it.pop(done)

#             try:
#                 yield await done
#             except StopAsyncIteration:
#                 continue

#             task = asyncio.create_task(it.__anext__())
#             task_to_it[task] = it
