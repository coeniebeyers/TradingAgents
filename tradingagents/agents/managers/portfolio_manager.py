"""Portfolio Manager: turns the Research Manager's rating and the risk debate into the plan.

The rating is the Research Manager's, the run's ``final_rating``: the research
view of the instrument. The Portfolio Manager decides how to act on it in this
book (size, entry, levels, horizon, funding) and cannot change it, since a
planning constraint such as an empty book is not a judgement on the stock.
Uses LangChain's ``with_structured_output`` for a typed ``PortfolioDecision``,
rendered with the rating to markdown as ``final_trade_decision`` for the memory
log, CLI display and saved reports; without structured output, the free text
is used under the same rating line.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_events_context_from_state,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.portfolio import SIZING_HEADER


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        events_context = get_events_context_from_state(state)
        events_line = (
            f"{events_context}Put timing and sizing around these events in the Executive Summary.\n"
            if events_context
            else ""
        )

        rating = state.get("research_rating") or RATING_REVIEW
        company = state.get("company_of_interest", "this instrument")
        if rating == RATING_REVIEW:
            rating_line = (f"**The rating**: the Research Manager's rating for {company} could not be read, so the run"
                           " is marked for review. Plan for keeping the current position unchanged until it is.")
        else:
            rating_line = (f"**The rating is decided**: the Research Manager rated {company} **{rating}**. That is the"
                           " research view of the instrument and it is not yours to change; your job is how to act on"
                           " it in this portfolio.")

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # A stage-2 plan comes with a sizing brief computed from the ratings and the book: the size is settled,
        # and the plan is how and when to get there.
        if SIZING_HEADER in portfolio_context:
            sizing_line = ("\n**Sizing is decided**: the portfolio block's sizing brief sets this position's target weight"
                           " and how the book funds a change. Your job is the action, the entry timing (the market and"
                           " sentiment reports and the scheduled events are for this), the levels, and the funding"
                           " steps. You may stage or delay a change for timing and must say why; do not change the"
                           " target weight.\n")
            output = """- **Action**: one of Open / Add / Hold / Trim / Exit / Stay out
- **Target Weight**: the brief's target, in percent of the book
- **Entry**: how and when to act (now, staged over a stated period, on a pullback to a level, after an event)
- **Levels**: the prices at which to add, trim or exit
- **Funding**: where the money comes from, or what a trim frees up
- **Executive Summary**: the plan in two to four sentences
- **Investment Thesis**: the evidence the plan rests on, and what would change it"""
        else:
            sizing_line = ""
            output = """- **Executive Summary**: the plan and how to act on it
- **Investment Thesis**: the evidence the plan rests on, and what would change it"""

        prompt = f"""As the Portfolio Manager, turn the research view into the plan for this position, using the risk analysts' debate.

{instrument_context}

{portfolio_context}

---

{rating_line}
{sizing_line}
**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{events_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts. Plan for the actual book: the position size, the entry (now, staged, or on a pullback or confirmation), the levels at which to add, trim or exit, and the time horizon. Where the plan cannot be carried out as the rating implies, such as an add with no cash, say how to fund it (what to trim) or what to wait for: that is a planning constraint, not a reason to change the rating. The risk debate always contains conflicting stances; weigh them on their merits, independent of speaking order.

## Output

Write these sections, in this order:

{output}

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = render_pm_decision(decision, rating)
        else:
            final_trade_decision = f"**Rating**: {rating}\n\n{llm.invoke(prompt).content}"

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_rating": rating,
        }

    return portfolio_manager_node
