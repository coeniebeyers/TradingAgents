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
