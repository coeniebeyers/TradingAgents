"""The caller's calendar of scheduled events reaches the steps that size and time a position.

A pending earnings report or vote is a reason to stage an entry, not to change the view, so the risk debaters and the
Portfolio Manager get the calendar, the Research Manager is told timing is theirs, and no calendar changes nothing.
"""

from __future__ import annotations

import importlib
import json

import pytest

from tests.test_portfolio_context import _bare_graph

DECIDERS = [
    ("tradingagents.agents.managers.portfolio_manager", "create_portfolio_manager"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
]


def _prompt(module, factory, events=None):
    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else json.dumps(str(prompt)))
            from langchain_core.messages import AIMessage
            return AIMessage("Recommendation: Hold\n\nnothing to do")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError  # force the free-text path

    state = {
        "company_of_interest": "GOOGL", "trade_date": "2026-10-27", "asset_type": "stock",
        "instrument_context": "", "market_report": "M", "sentiment_report": "S",
        "news_report": "N", "fundamentals_report": "F", "investment_plan": "P",
        "trader_investment_plan": "T", "past_context": "", "portfolio_context": "",
        "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "",
                                    "current_response": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
    }
    if events is not None:
        state["events"] = events
    getattr(importlib.import_module(module), factory)(_LLM())(state)
    return seen[0]


CALENDAR = "- Earnings date: 2026-10-28 (consensus EPS 3.33)"


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", DECIDERS)
def test_risk_team_and_portfolio_manager_size_around_the_calendar(module, factory):
    text = _prompt(module, factory, CALENDAR)
    assert CALENDAR in text and "size and time the position" in text


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", DECIDERS + [
    ("tradingagents.agents.managers.research_manager", "create_research_manager")])
def test_prompts_are_unchanged_without_a_calendar(module, factory):
    assert _prompt(module, factory) == _prompt(module, factory, "") == _prompt(module, factory, "  \n")


@pytest.mark.unit
def test_portfolio_manager_keeps_the_rating_on_the_research_view():
    text = _prompt(*DECIDERS[0], CALENDAR)
    assert "a scheduled event alone is not a reason to change the rating" in text


@pytest.mark.unit
def test_research_manager_leaves_timing_to_the_risk_team_but_never_sees_the_calendar():
    module = ("tradingagents.agents.managers.research_manager", "create_research_manager")
    text = _prompt(*module, CALENDAR)
    assert "decided later by the risk team" in text and CALENDAR not in text


@pytest.mark.unit
def test_the_calendar_is_part_of_the_run_state_and_its_checkpoint_signature(tmp_path):
    graph = _bare_graph(tmp_path)
    assert graph.create_run_state("GOOGL", "2026-10-27", events=CALENDAR)["events"] == CALENDAR
    assert graph.create_run_state("GOOGL", "2026-10-27")["events"] == ""
    assert graph._run_signature("stock") == graph._run_signature("stock", events="")
    assert graph._run_signature("stock") != graph._run_signature("stock", events=CALENDAR)
