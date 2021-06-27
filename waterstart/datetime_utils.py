import datetime


def get_midnight(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time.min)


def to_timedelta(dt: datetime.datetime) -> datetime.timedelta:
    return dt - datetime.datetime.min.replace(tzinfo=dt.tzinfo)


def delta_to_midnight(dt: datetime.datetime) -> datetime.timedelta:
    # return dt - get_midnight(dt).replace(tzinfo=dt.tzinfo)
    return (
        datetime.datetime.combine(datetime.date.min, dt.time())
        - datetime.datetime.min
    )
