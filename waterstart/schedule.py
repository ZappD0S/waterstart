import datetime
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
from zoneinfo import ZoneInfo

from .openapi import ProtoOAHoliday
from .symbols import SymbolInfoWithConvChains


def get_midnight(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time.min)


def to_timedelta(dt: datetime.datetime) -> datetime.timedelta:
    return dt - datetime.datetime.min


# TODO: we need the ability to get also the last valid time
# TODO: we don't really need the classes below to be dataclasses, so let's remove that


class BaseSchedule(ABC):
    @abstractmethod
    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        ...

    @abstractmethod
    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        ...


class ScheduleCombinator(BaseSchedule):
    def __init__(self, schedules: Sequence[BaseSchedule]):
        self._schedules = schedules

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        new_dt = dt

        while True:
            for schedule in self._schedules:
                new_dt = schedule.last_valid_time(new_dt)

            if new_dt == dt:
                return dt

            dt = new_dt

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        new_dt = dt

        while True:
            for schedule in self._schedules:
                new_dt = schedule.next_valid_time(new_dt)

            if new_dt == dt:
                return dt

            dt = new_dt


@dataclass(frozen=True)
class Holiday(BaseSchedule):
    start: datetime.datetime
    end: datetime.datetime
    timezone: datetime.tzinfo = field(init=False)
    is_recurring: bool

    def __post_init__(self):
        if self.start.tzinfo is None or self.start.tzinfo != self.start.tzinfo:
            raise ValueError()

        super().__setattr__("timezone", self.start.tzinfo)

        if self.start.date() != self.end.date():
            raise ValueError()

        if self.start >= self.end:
            raise ValueError()

    @property
    def year(self) -> int:
        return self.start.year

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        ...

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        original_tz = dt.tzinfo
        original_year = dt.year

        dt = dt.astimezone(self.timezone)

        if self.is_recurring:
            try:
                dt = dt.replace(year=self.year)
            except ValueError:
                # NOTE: it's probably february 29th and the recurring
                #  holiday was set on a non-leap year so it's safe to
                # assume this it's not a holiday
                return dt.astimezone(original_tz)

        times = [self.start, self.end]
        index = bisect_right(times, dt)

        if index % 2 == 0:
            return dt.replace(year=original_year).astimezone(original_tz)
        else:
            return times[index].replace(year=original_year).astimezone(original_tz)


# TODO: abtract class that combines multiple schedules (called ScheduleCombinator, or CompositeSchedule or ...)


# TODO: This is going to be WeekSchedule and not have the holidays
# TODO Then create a SymbolSchedule that combines the week and multiple holiday schedules
@dataclass(frozen=True)
class SymbolSchedule:
    timetable: Sequence[datetime.timedelta]
    timezone: datetime.tzinfo
    holidays: Sequence[Holiday]

    def __post_init__(self):
        if not self.timetable:
            raise ValueError()

        if len(self.timetable) % 2 != 0:
            raise ValueError()

        if not all(a < b for a, b in zip(self.timetable[:-1], self.timetable[1:])):
            raise ValueError()

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        original_tz = dt.tzinfo
        dt = dt.astimezone(self.timezone)

        # TODO: we need to apply holiday shift and first schedule start shift repeatedly until the result doesn't change

        for holiday in self.holidays:
            dt = holiday.next_valid_time(dt)

        days_since_last_sunday = (dt.weekday() + 1) % 7
        last_sunday = dt.date() - datetime.timedelta(days=days_since_last_sunday)
        last_sunday_midnight = get_midnight(last_sunday)

        delta_to_lsm = dt - last_sunday_midnight

        index = bisect_right(self.timetable, delta_to_lsm)
        offset: datetime.timedelta = datetime.timedelta.min

        if index % 2 == 0:
            first_trading_time_index = index + 1
            if first_trading_time_index == len(self.timetable):
                offset += datetime.timedelta(weeks=1)
                first_trading_time_index = 0

            first_trading_time = (
                last_sunday_midnight + offset + self.timetable[first_trading_time_index]
            )
        else:
            first_trading_time = dt

        return first_trading_time.astimezone(original_tz)


# TODO: this needs to listen to the symbol from SymbolList for the symbols that change
# and update consequently
class ExecutionSchedule:
    def __init__(
        self,
        symbols: Sequence[SymbolInfoWithConvChains],
        trading_interval: datetime.timedelta,
    ) -> None:
        self._symbols = symbols
        self._sym_schedule_map = {sym: self._get_sym_schedule(sym) for sym in symbols}
        self.trading_interval = trading_interval
        self._offset: datetime.timedelta = datetime.timedelta.min
        self._offset_date: datetime.date = datetime.date.min

    @property
    def traded_symbols(self) -> Sequence[SymbolInfoWithConvChains]:
        return self._symbols

    @staticmethod
    def _get_sym_schedule(sym_info: SymbolInfoWithConvChains) -> SymbolSchedule:
        def get_holiday(holiday: ProtoOAHoliday) -> Holiday:
            tz = ZoneInfo(holiday.scheduleTimeZone)
            epoch = datetime.datetime.fromtimestamp(0, tz)

            start = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.startSecond
            )
            end = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.endSecond
            )
            return Holiday(start, end, holiday.isRecurring)

        holidays = [get_holiday(holiday) for holiday in sym_info.symbol.holiday]

        times: list[datetime.timedelta] = []
        for interval in sym_info.symbol.schedule:
            start = datetime.timedelta(seconds=interval.startSecond)
            end = datetime.timedelta(seconds=interval.endSecond)
            times += [start, end]

        tz = ZoneInfo(sym_info.symbol.scheduleTimeZone)
        return SymbolSchedule(times, tz, holidays)

    def _get_offset(self, dt: datetime.datetime) -> datetime.timedelta:
        if (date := dt.date()) != self._offset_date:
            midnight = get_midnight(date)
            first_valid_time = self.next_valid_time(midnight)

            self._offset_date = date
            self._offset = to_timedelta(first_valid_time) % self.trading_interval

        return self._offset

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return max(
            schedule.next_valid_time(dt) for schedule in self._sym_schedule_map.values()
        )

    def _round_to_next_trading_perion(self, dt: datetime.datetime) -> datetime.datetime:
        offset = self._get_offset(dt)
        shifted_dt = dt + self.trading_interval

        return shifted_dt - to_timedelta(shifted_dt) % self.trading_interval + offset

    def next_trading_time(
        self, dt: Optional[datetime.datetime] = None
    ) -> datetime.datetime:

        if dt is None:
            dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)

        # dt = self._round_to_next_trading_perion(dt)

        # while (next_valid_time := self.next_valid_time(dt)) != dt:
        #     dt = self._round_to_next_trading_perion(next_valid_time)

        while True:
            dt = self._round_to_next_trading_perion(dt)
            new_dt = self.next_valid_time(dt)

            if new_dt == dt:
                break

            dt = new_dt

        return dt
