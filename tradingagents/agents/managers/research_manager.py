"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

import re

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.rating import RATING_REVIEW, parse_rating
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

_RECOMMENDATION = re.compile(r"^\W*recommendation\W*:\W*(buy|overweight|hold|underweight|sell)\b", re.I | re.M)


def research_rating(plan: str) -> str:
    """The Research Manager's rating, which is the run's: its labelled recommendation, else the
    rating its free text opens with or labels, else REVIEW."""
    m = _RECOMMENDATION.search(plan or "")
    return m.group(1).capitalize() if m else parse_rating(plan or "", RATING_REVIEW)


def create_research_manager(llm):
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

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not manufacture a direction to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.{timing}

---

{facts}**Debate History:**
{history}

## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments decided it
- **Strategic Actions**: concrete steps for the trader, sized against a standard allocation

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

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
            "research_rating": research_rating(investment_plan),
        }

    return research_manager_node
