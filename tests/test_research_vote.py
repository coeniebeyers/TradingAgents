"""The Research Manager's vote: several answers to one debate, rated by their majority."""

import pytest

from tradingagents.agents.managers.research_manager import voted_plan
from tradingagents.agents.rating import run_rating, vote_rating


@pytest.mark.unit
@pytest.mark.parametrize("ratings, expected", [
    (["Underweight", "Underweight", "Hold"], "Underweight"),
    (["Overweight", "Hold", "Underweight"], "Hold"),      # no majority: the evidence is split
    (["Buy", "Overweight", "Overweight"], "Overweight"),
    (["REVIEW", "Sell", "Sell"], "Sell"),
    (["REVIEW", "Buy", "Hold"], "Hold"),                  # one each of two rated answers
    (["REVIEW", "REVIEW"], "REVIEW"),
])
def test_vote_is_the_majority_else_hold(ratings, expected):
    assert vote_rating(ratings) == expected


def _plan(rating: str, why: str) -> str:
    return f"**Recommendation**: {rating}\n\n**Rationale**: {why}"


@pytest.mark.unit
def test_voted_plan_is_an_answer_that_agrees_with_the_vote():
    plan, rating = voted_plan([_plan("Hold", "a"), _plan("Underweight", "b"), _plan("Underweight", "c")])
    assert rating == "Underweight" and plan == _plan("Underweight", "b")


@pytest.mark.unit
def test_a_split_without_a_hold_answer_keeps_every_answer():
    plan, rating = voted_plan([_plan("Overweight", "a"), _plan("Underweight", "b"), _plan("Buy", "c")])
    assert rating == "Hold"
    assert plan.startswith("**Recommendation**: Hold")
    assert all(why in plan for why in ("Rationale**: a", "Rationale**: b", "Rationale**: c"))


@pytest.mark.unit
def test_a_research_run_is_rated_by_its_research_manager():
    assert run_rating({"research_rating": "Underweight"}) == "Underweight"
    assert run_rating({}) == "REVIEW"
    assert run_rating({"final_rating": "Hold", "research_rating": "Underweight"}) == "Hold"


# GOOGL's three answers on 2026-10-29 were all of this shape: a long hidden reasoning, then only the verdict.
_THIN = ("**Recommendation**: Hold\n\n**What the price assumes**: The price assumes a very demanding FCF recovery\n\n"
         "**Variant view**: None\n\n**Margin of safety**: None\n\n**Rationale**: Strong growth versus collapsed FCF; "
         "evidence too thin for mispricing.\n\n**Strategic Actions**: Hold at standard allocation.")
_FULL = ("**Recommendation**: Hold\n\n**What the price assumes**: FCF growing +41.0% a year for 10 years at an 11.1% "
         "discount rate, from a 0.6% FCF yield.\n\n**Variant view**: None\n\n**Margin of safety**: None\n\n"
         "**Rationale**: " + "Revenue +16% and operating income +30% support the bull case; capex at 104% of operating "
         "cash flow and a 0.6% yield support the bear's; the bear's cash-burn point stands, the bull's moat claim is "
         "unproven, so the price is about right for now and neither side shows a mispricing with a margin of safety."
         + "\n\n**Strategic Actions**: Standard allocation.")


@pytest.mark.unit
def test_an_answer_with_only_its_verdict_is_thin():
    from tradingagents.agents.managers.research_manager import thin_reasons

    assert thin_reasons(_THIN) == ["'What the price assumes' gives no numbers",
                                   "the rationale is a line, not the evidence for and against"]
    assert thin_reasons(_FULL) == []
    assert "'What the price assumes' names a rating" in thin_reasons(_FULL.replace("FCF growing", "Hold: FCF growing"))


@pytest.mark.unit
def test_an_answer_under_headings_is_read_like_a_labelled_one():
    from tradingagents.agents.managers.research_manager import thin_reasons

    headed = _FULL.replace("**What the price assumes**: ", "## What the price assumes\n\n").replace(
        "**Rationale**: ", "## Rationale\n\n")
    assert thin_reasons(headed) == []


@pytest.mark.unit
def test_a_thin_answer_is_asked_again_once_with_what_it_lacked():
    from unittest.mock import MagicMock

    from tradingagents.agents.managers.research_manager import create_research_manager

    prompts = []
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("free text")
    llm.invoke.side_effect = lambda p: prompts.append(p) or MagicMock(content=_THIN if len(prompts) == 1 else _FULL)
    state = {"company_of_interest": "GOOGL", "trade_date": "2026-10-29",
             "investment_debate_state": {"history": "bull and bear", "count": 2}}
    out = create_research_manager(llm)(state)
    assert len(prompts) == 2 and "too thin to act on: 'What the price assumes' gives no numbers" in prompts[1]
    assert out["investment_plan"] == _FULL


@pytest.mark.unit
def test_a_full_answer_is_not_asked_again():
    from unittest.mock import MagicMock

    from tradingagents.agents.managers.research_manager import create_research_manager

    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("free text")
    llm.invoke.return_value = MagicMock(content=_FULL)
    state = {"company_of_interest": "GOOGL", "trade_date": "2026-10-29",
             "investment_debate_state": {"history": "bull and bear", "count": 2}}
    create_research_manager(llm)(state)
    assert llm.invoke.call_count == 1
