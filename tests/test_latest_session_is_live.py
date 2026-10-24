"""A run dated to the latest closed session is live, not historical.

Treating every run not dated today as a backtest withheld profiles, statements,
insider trades and prediction markets from every analysis of the last close.
"""
from datetime import date

import pytest

from tradingagents.dataflows.date_window import is_historical

FRI, SAT, SUN, MON, TUE = (date(2026, 10, d) for d in (23, 24, 25, 26, 27))


@pytest.mark.unit
@pytest.mark.parametrize("run, today, historical", [
    ("2026-10-23", FRI, False),   # same day
    ("2026-10-23", SAT, False),   # Friday's close, seen on Saturday
    ("2026-10-23", MON, False),   # Friday's close, seen on Monday before anything new closed
    ("2026-10-23", TUE, True),    # Monday has traded since
    ("2026-10-22", SAT, True),    # Thursday, with Friday since
    ("2026-10-27", MON, False),   # a future date is never historical
    ("", MON, False),
])
def test_historical_means_a_session_traded_since(run, today, historical):
    assert is_historical(run, today=today) is historical
