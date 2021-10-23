from __future__ import annotations
from abc import ABC, abstractmethod

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Optional, TYPE_CHECKING, TypeVar, Union
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from _typeshed import SupportsLessThan


def is_sorted(seq: Sequence[SupportsLessThan]) -> bool:
    return all(a < b for a, b in zip(seq[:-1], seq[1:]))


def is_contiguous(seq: Sequence[int]) -> bool:
    return all(b - a == 1 for a, b in zip(seq[:-1], seq[1:]))


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class ComposableAsyncIterable(ABC, AsyncIterable[T]):
    @staticmethod
    @asynccontextmanager
    async def from_it(
        iterable: AsyncIterable[U],
    ) -> AsyncIterator[ComposableAsyncIterable[U]]:
        comp_it = SingleComposableAsyncIterable(iterable)
        try:
            yield comp_it
        finally:
            await comp_it.aclose()

    @abstractmethod
    def __or__(
        self, other: ComposableAsyncIterable[U]
    ) -> ComposableAsyncIterable[Union[T, U]]:
        ...

    @abstractmethod
    def _get_tasks(self) -> tuple[asyncio.Task[T], ...]:
        ...


class SingleComposableAsyncIterable(ComposableAsyncIterable[T]):
    def __init__(self, iterable: AsyncIterable[T]) -> None:
        it = iterable.__aiter__()
        self._wrapped_it = it
        self._task: Optional[asyncio.Task[T]] = None
        self._it = self._get_iterator()

    def __or__(
        self, other: ComposableAsyncIterable[U]
    ) -> ComposableAsyncIterable[Union[T, U]]:
        return MultipleComposableAsyncIterable(self, other)  # type: ignore

    def __aiter__(self) -> AsyncIterator[T]:
        return self._it

    async def _get_iterator(self) -> AsyncIterator[T]:
        it = self._wrapped_it

        while True:
            if (task := self._task) is None:
                yield await it.__anext__()
            else:
                yield await task
                self._task = None

    def _get_tasks(self) -> tuple[asyncio.Task[T], ...]:
        if (task := self._task) is None:
            task = self._task = asyncio.create_task(
                self._wrapped_it.__anext__(), name="comp_it_next"
            )

        if task.done() and task.exception() is None:
            self._task = asyncio.create_task(
                self._wrapped_it.__anext__(), name="comp_it_next"
            )

        return (task,)

    async def aclose(self) -> None:
        if (task := self._task) is None:
            return

        task.cancel()

        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass


class MultipleComposableAsyncIterable(ComposableAsyncIterable[Union[T, U]]):
    def __init__(
        self, first: ComposableAsyncIterable[T], second: ComposableAsyncIterable[U]
    ) -> None:
        self._first = first
        self._second = second
        self._it = self._get_iterator()

    def _get_tasks(self) -> tuple[asyncio.Task[Union[T, U]], ...]:
        return self._first._get_tasks() + self._second._get_tasks()  # type: ignore

    def __or__(
        self, other: ComposableAsyncIterable[V]
    ) -> ComposableAsyncIterable[Union[T, U, V]]:
        return MultipleComposableAsyncIterable(self, other)  # type: ignore

    def __aiter__(self) -> AsyncIterator[Union[T, U]]:
        return self._it

    async def _get_iterator(self) -> AsyncIterator[Union[T, U]]:
        while True:
            done_tasks, _ = await asyncio.wait(
                self._get_tasks(), return_when=asyncio.FIRST_COMPLETED
            )

            done = True

            for task in done_tasks:
                try:
                    yield await task
                except StopAsyncIteration:
                    pass
                else:
                    done = False

            if done:
                break
