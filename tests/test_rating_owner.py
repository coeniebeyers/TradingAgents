"""The Research Manager rates; the Portfolio Manager plans.

With an empty book, the Portfolio Manager turned five of thirty Overweights into Holds because an add "cannot be
executed": a planning constraint read as a judgement on the stock. The rating is now the Research Manager's, and the
debaters argue from the same code-computed fact sheet the Research Manager checks them against.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import research_rating
from tradingagents.agents.schemas import PortfolioDecision

RISK = {"history": "h", "aggressive_history": "", "conservative_history": "", "neutral_history": "",
        "current_aggressive_response": "", "current_conservative_response": "", "current_neutral_response": "",
        "latest_speaker": "Neutral", "count": 1}


def _pm(rating, structured=True, answer="Hold the position; add on a pullback."):
    captured = {}
    llm = MagicMock()
    if structured:
        bound = MagicMock()
        bound.invoke.side_effect = lambda p: captured.setdefault("prompt", p) and PortfolioDecision(
            executive_summary=answer, investment_thesis="t")
        llm.with_structured_output.return_value = bound
    else:
        llm.with_structured_output.side_effect = NotImplementedError
        llm.invoke.side_effect = lambda p: captured.setdefault("prompt", p) and MagicMock(content=answer)
    state = {"company_of_interest": "BRK-B", "risk_debate_state": RISK, "investment_plan": "plan",
             "trader_investment_plan": "trader plan", "portfolio_context": "0.00 USD cash"}
    if rating is not None:
        state["research_rating"] = rating
    out = create_portfolio_manager(llm)(state)
    return out, captured["prompt"]


@pytest.mark.unit
@pytest.mark.parametrize("plan, rating", [
    ("**Recommendation**: Overweight\n\n**Rationale**: the bear case on Hold-worthy valuation lost.", "Overweight"),
    ("Underweight\n\nThe bear case carried it.", "Underweight"),          # free text opening with the rating
    ("Rationale: ... Strategic Actions: ...", "REVIEW"),                   # the empty template seen in run 178
    ("## What the price assumes\n\nHold-worthy? 28% FCF growth.\n\n## Recommendation\n\nHold\n", "Hold"),  # run 239
    ("## Recommendation\n\nThe bull case on Overweight lost.", "REVIEW"),  # a heading over prose is no call
])
def test_the_research_manager_s_rating_is_read_from_its_plan(plan, rating):
    assert research_rating(plan) == rating


@pytest.mark.unit
@pytest.mark.parametrize("structured", [True, False])
def test_the_portfolio_manager_plans_around_the_rating_and_cannot_change_it(structured):
    out, prompt = _pm("Overweight", structured, answer="Rating: Hold. No cash, so do not add.")
    assert out["final_rating"] == "Overweight"
    assert out["final_trade_decision"].startswith("**Rating**: Overweight")
    assert "the Research Manager rated BRK-B **Overweight**" in prompt
    assert "a planning constraint, not a reason to change the rating" in prompt
    assert "Rating Scale" not in prompt


@pytest.mark.unit
def test_an_unreadable_research_rating_stays_review():
    out, prompt = _pm(None)
    assert out["final_rating"] == "REVIEW" and "could not be read" in prompt


@pytest.mark.unit
def test_the_portfolio_manager_s_schema_has_no_rating_to_give():
    assert "rating" not in PortfolioDecision.model_fields


def _debater_prompt(side, fact_sheet=None):
    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else json.dumps(str(prompt)))
            return MagicMock(content="argument")

    state = {"company_of_interest": "META", "asset_type": "stock", "instrument_context": "",
             "market_report": "M", "sentiment_report": "S", "news_report": "N", "fundamentals_report": "F",
             "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "",
                                         "current_response": "", "count": 0}}
    if fact_sheet is not None:
        state["fact_sheet"] = fact_sheet
    module = importlib.import_module(f"tradingagents.agents.researchers.{side}_researcher")
    getattr(module, f"create_{side}_researcher")(_LLM())(state)
    return seen[0]


@pytest.mark.unit
@pytest.mark.parametrize("side", ["bull", "bear"])
def test_the_debaters_argue_from_the_fact_sheet(side):
    text = _debater_prompt(side, "- Diluted EPS 3.83; vs 1.05 a year earlier (+264.8% year on year)")
    assert "+264.8% year on year" in text and "the fact sheet is right" in text
    assert _debater_prompt(side) == _debater_prompt(side, "") and "Fact sheet" not in _debater_prompt(side)


@pytest.mark.unit
def test_with_a_sizing_brief_the_portfolio_manager_plans_the_timing_not_the_size():
    from tradingagents.agents.schemas import PlanAction, render_pm_decision
    from tradingagents.portfolio import SIZING_HEADER

    captured = {}
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.side_effect = lambda p: captured.setdefault("prompt", p) and PortfolioDecision(
        executive_summary="Trim to 5% over two weeks.", investment_thesis="t", action=PlanAction.TRIM,
        target_weight_pct=5, entry="staged over two weeks", levels="trim faster above 80", funding="frees about 7,000")
    llm.with_structured_output.return_value = bound
    state = {"company_of_interest": "PYPL", "risk_debate_state": RISK, "investment_plan": "plan",
             "trader_investment_plan": "trader plan", "research_rating": "Underweight",
             "portfolio_context": f"Portfolio at the analysis date:\n- Current position\n\n{SIZING_HEADER}\nTarget 5.0%"}
    out = create_portfolio_manager(llm)(state)
    assert "**Sizing is decided**" in captured["prompt"] and "do not change the target weight" in captured["prompt"]
    assert "- **Funding**: where the money comes from" in captured["prompt"]
    text = out["final_trade_decision"]
    assert "**Action**: Trim" in text and "**Target Weight**: 5% of the book" in text
    assert "**Funding**: frees about 7,000" in text and out["final_rating"] == "Underweight"
    plain = render_pm_decision(PortfolioDecision(executive_summary="e", investment_thesis="t"), "Hold")
    assert not any(f"**{label}**" in plain for label in ("Action", "Target Weight", "Entry", "Levels", "Funding"))


@pytest.mark.unit
def test_without_a_sizing_brief_the_prompt_is_unchanged():
    _, prompt = _pm("Hold")
    assert "Sizing is decided" not in prompt and "- **Action**" not in prompt
