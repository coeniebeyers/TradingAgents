"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.rating import RATING_REVIEW, parse_rating, vote_rating
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

_RECOMMENDATION = re.compile(r"^\W*recommendation\W*:\W*(buy|overweight|hold|underweight|sell)\b", re.I | re.M)
# A free-text answer that ends on a "## Recommendation" heading with the rating alone below it.
_RECOMMENDATION_HEADING = re.compile(
    r"^#+\s*\W*recommendation\W*\s*\n\s*\W*(buy|overweight|hold|underweight|sell)\W*\s*$", re.I | re.M)


# A section of an answer, labelled ("**Rationale**: ...") or under a heading ("## Rationale").
_SECTION = re.compile(r"(?:\*\*|^#+[ \t]*)(What the price assumes|Rationale)(?:\*\*[ \t]*:|[ \t]*\n)\s*(.*?)(?=\n\*\*|\n#|\Z)",
                      re.M | re.S)
_RATING_WORD = re.compile(r"\b(Buy|Overweight|Hold|Underweight|Sell)\b")
_MIN_RATIONALE = 300


def thin_reasons(plan: str) -> list[str]:
    """Why an answer is too thin to act on, if it is.

    The model reasons at length before it answers and sometimes keeps only the verdict: "the price assumes a very
    demanding recovery; Hold" leaves the reader nothing to check or act on, and a vote of such answers agrees on
    nothing."""
    sections = {name: text.strip() for name, text in _SECTION.findall(plan or "")}
    implied, rationale = sections.get("What the price assumes", ""), sections.get("Rationale", "")
    reasons = []
    if not re.search(r"\d", implied):
        reasons.append("'What the price assumes' gives no numbers")
    if _RATING_WORD.search(implied):
        reasons.append("'What the price assumes' names a rating")
    if len(rationale) < _MIN_RATIONALE:
        reasons.append("the rationale is a line, not the evidence for and against")
    return reasons


def research_rating(plan: str) -> str:
    """The Research Manager's rating, which is the run's: its labelled recommendation or recommendation
    heading, else the rating its free text opens with or labels, else REVIEW."""
    m = _RECOMMENDATION.search(plan or "") or _RECOMMENDATION_HEADING.search(plan or "")
    return m.group(1).capitalize() if m else parse_rating(plan or "", RATING_REVIEW)


def voted_plan(plans: list[str]) -> tuple[str, str]:
    """The plan and rating of several answers to the same debate: the first answer that agrees with
    the vote, or, when the vote is a Hold no answer gave, a note of the split above every answer."""
    ratings = [research_rating(p) for p in plans]
    rating = vote_rating(ratings)
    for plan, r in zip(plans, ratings, strict=True):
        if r == rating:
            return plan, rating
    answers = "\n\n---\n\n".join(f"**Answer {i}** ({r})\n\n{p}" for i, (p, r) in enumerate(zip(plans, ratings, strict=True), 1))
    return (f"**Recommendation**: {rating}\n\n**Rationale**: The {len(plans)} answers split "
            f"({', '.join(ratings)}) with no majority, so the evidence is split: {rating}.\n\n---\n\n{answers}"), rating


def create_research_manager(llm, samples: int = 1):
    """The Research Manager node. With ``samples`` above 1 it answers that many times side by side
    and the rating is their vote (see ``voted_plan``)."""
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        fact_sheet = (state.get("fact_sheet") or "").strip()
        # Debaters repeat an analyst's misread table (a 69% EPS fall the filings show as +265%), and this step sees
        # only the debate, so figures computed by code from the same data are put beside it when the caller has them.
        facts = (f"""**Fact sheet** (computed by code from the data the analysts retrieved; where the debate disagrees with it, the fact sheet is right):

{fact_sheet}

---

""" if fact_sheet else "")
        # Event timing is the risk team's: they get the calendar, so the rating rests on the investment case.
        timing = (" Timing around scheduled events (an earnings date, a vote, a macro release) is decided later by the"
                  " risk team, which has the calendar; rate the investment case, not the timing."
                  if (state.get("events") or "").strip() else "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager, judge this debate and decide whether the market misprices this instrument over the next 3 to 5 years.

{instrument_context}

---

**Rating Scale** (a verdict on mispricing over 3-5 years; use exactly one):
- **Buy**: clearly undervalued, with a strong, well-evidenced variant view and a wide margin of safety. Rare.
- **Overweight**: likely undervalued: a credible variant view on the fundamentals, with a margin of safety.
- **Hold**: priced about right for its fundamentals, or the evidence is too thin to say otherwise. The default.
- **Underweight**: likely overvalued: the price assumes more than the fundamentals plausibly deliver.
- **Sell**: clearly overvalued or fundamentally impaired, well evidenced. Rare.

Start from Hold: the market prices most stocks about right, and a rating off Hold claims it has this one wrong. Move off Hold only when you can state what the price assumes, where the fundamental evidence says otherwise, the margin of safety, and what would prove you wrong; if you cannot, the answer is Hold. A Hold takes the same work as any other rating: state what the price assumes and which evidence supports it and which cuts against it; "mixed fundamentals" alone is not a reason. Judge on fundamentals: what the price assumes (the fact sheet's reverse discounted cash flow), the business's record of growth, margins and returns on capital, its balance sheet and cash generation, and material news about the business. Price trends, technical indicators and social-media sentiment are for timing an entry, which is decided later; they are not evidence of mispricing and must not move the rating. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.{timing}

---

{facts}**Debate History:**
{history}

## Output

Reason before you rate. Write these sections, in this order, and give the recommendation last:

- **What the price assumes**: the growth, margins or returns the price implies over 3-5 years, in numbers (the reverse discounted cash flow where the fact sheet has one). Never a rating.
- **Variant view**: where the evidence says the market is wrong, with that evidence, or "None"
- **Margin of safety**: how far the price is from a supported value, or "None"
- **What would prove it wrong**: observable facts
- **Rationale**: which fundamental evidence decided it, and which debate arguments you accepted or rejected
- **Strategic Actions**: how to hold the position over the horizon, against a standard allocation
- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell, following from the sections above

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        def answer() -> str:
            plan = invoke_structured_or_freetext(structured_llm, llm, prompt, render_research_plan, "Research Manager")
            reasons = thin_reasons(plan)
            if not reasons:
                return plan
            # Asked once more, with what was missing; the fuller of the two answers stands.
            again = invoke_structured_or_freetext(
                structured_llm, llm,
                prompt + f"""

---

A first answer to this was too thin to act on: {"; ".join(reasons)}. Answer again in full: the numbers the price implies, the fundamental evidence for and against, and which debate arguments you accepted or rejected.""",
                render_research_plan, "Research Manager")
            return again if len(thin_reasons(again)) <= len(reasons) else plan

        if samples > 1:
            # Each answer runs in a copy of this context, so the run's callbacks see its model calls.
            contexts = [contextvars.copy_context() for _ in range(samples)]
            with ThreadPoolExecutor(samples) as pool:
                plans = list(pool.map(lambda ctx: ctx.run(answer), contexts))
            investment_plan, rating = voted_plan(plans)
        else:
            plans = []
            investment_plan = answer()
            rating = research_rating(investment_plan)

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "research_rating": rating,
            "research_samples": plans,
        }

    return research_manager_node
