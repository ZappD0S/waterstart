import datetime
from bisect import bisect_right
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

from .openapi import ProtoOAHoliday
from .symbols import SymbolInfo


@dataclass
class Holiday:
    # date: datetime.date
    start: datetime.datetime
    end: datetime.datetime
    timezone: datetime.tzinfo
    is_recurring: bool

    def __post_init__(self):
        if self.start.date() != self.end.date():
            raise ValueError()

        if self.start >= self.end:
            raise ValueError()

    @property
    def year(self):
        return self.start.year

    def next_trading_time(self, dt: datetime.datetime) -> datetime.datetime:
        original_tz = dt.tzinfo
        original_year = dt.year

        dt = dt.astimezone(self.timezone)

        if self.is_recurring:
            try:
                dt = dt.replace(year=self.year)
            except ValueError:
                # TODO: warn..
                return dt.astimezone(original_tz)

        times = [self.start, self.end]
        index = bisect_right(times, dt)

        dt = dt.replace(year=original_year, tzinfo=original_tz)

        if index % 2 == 0:
            return dt.replace(year=original_year).astimezone(original_tz)
        else:
            return times[index].replace(year=original_year).astimezone(original_tz)

        # return (
        #     self.date.day == date.day
        #     and self.date.month == date.month
        #     and (self.is_recurring or self.date.year == date.year)
        # )


@dataclass
class Schedule:
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

    # def is_holiday(self, date: datetime.date) -> bool:
    #     return any(holiday.is_holiday(date) for holiday in self.holidays)

    def next_trading_time(self, dt: datetime.datetime) -> datetime.datetime:
        original_tz = dt.tzinfo
        dt = dt.astimezone(self.timezone)
        # date = dt.date()

        for holiday in self.holidays:
            dt = holiday.next_trading_time(dt)

        # while self.is_holiday(date):
        #     date += datetime.timedelta(days=1)

        # if date > dt.date():
        #     dt = datetime.datetime.combine(date, datetime.time.min)

        days_since_last_sunday = (dt.weekday() + 1) % 7
        last_sunday = dt.date() - datetime.timedelta(days=days_since_last_sunday)
        last_sunday_midnight = datetime.datetime.combine(last_sunday, datetime.time.min)

        time_since_lsm = dt - last_sunday_midnight

        index = bisect_right(self.timetable, time_since_lsm)
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


class ExecutionSchedule:
    def __init__(
        self, symbols: Sequence[SymbolInfo], trading_interval: datetime.timedelta
    ) -> None:
        self._shedule_map = {sym: self._get_schedule(sym) for sym in symbols}
        self.trading_interval = trading_interval

    @staticmethod
    def _get_schedule(sym_info: SymbolInfo) -> Schedule:
        # TODO: convert all datetimes to UTC so we don't need to do any conversions in the
        # other methods..

        def get_holiday(holiday: ProtoOAHoliday) -> Holiday:
            tz = ZoneInfo(holiday.scheduleTimeZone)
            start = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.startSecond
            )
            end = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.endSecond
            )
            return Holiday(start, end, tz, holiday.isRecurring)

        epoch = datetime.datetime.fromtimestamp(0)
        holidays = [get_holiday(holiday) for holiday in sym_info.symbol.holiday]

        times: list[datetime.timedelta] = []
        for interval in sym_info.symbol.schedule:
            start = datetime.timedelta(seconds=interval.startSecond)
            end = datetime.timedelta(seconds=interval.endSecond)
            times += [start, end]

        tz = ZoneInfo(sym_info.symbol.scheduleTimeZone)
        return Schedule(times, tz, holidays)

    # TODO: just return a time instead of an awaitable
    def wait_until_next_timestep(
        self, dt: Optional[datetime.datetime] = None
    ) -> Awaitable[None]:
        if dt is None:
            dt = datetime.datetime.now()

        next_trading_time = max(
            schedule.next_trading_time(dt) for schedule in self._shedule_map.values()
        )

        # TODO: round the value above to the next
        pass
