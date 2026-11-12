"""The Research Manager starts from the instrument's previous research record and needs new, material evidence to move.

Judged from scratch every run, a name near the line flipped between nights on a different draw of the same debate (CVS
split Overweight/Hold/Overweight on identical inputs, 2026-11-12). The caller can hand in the last record; without one
the prompt is exactly as before.
"""

from __future__ import annotations

import json

import pytest

from tests.test_portfolio_context import _bare_graph

PRIOR = ("As of 2026-11-02: Overweight (vote Overweight, Overweight, Overweight)\n"
         "What the price assumed: FCF after SBC growing +3.4% a year for 10 years\n"
         "What would prove it wrong: guidance below 3% growth")


def _prompt(prior=None):
    from tradingagents.agents.managers.research_manager import create_research_manager

    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else json.dumps(str(prompt)))
            from langchain_core.messages import AIMessage
            return AIMessage("Recommendation: Hold\n\nnothing to do")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError  # force the free-text path

    state = {"company_of_interest": "BKNG", "trade_date": "2026-11-03", "asset_type": "stock", "instrument_context": "",
             "investment_debate_state": {"history": "Bull: ... Bear: ...", "bull_history": "", "bear_history": "",
                                         "current_response": "", "count": 0}}
    if prior is not None:
        state["prior_record"] = prior
    create_research_manager(_LLM())(state)
    return seen[0]


@pytest.mark.unit
def test_with_a_previous_record_the_manager_starts_from_it_and_names_what_changed():
    text = _prompt(PRIOR)
    assert PRIOR in text and "Your previous research record" in text
    assert "Start from the previous record's rating" in text and "Start from Hold" not in text
    assert "newly reported results or filings, a change in guidance" in text
    assert "analysts' opinions and price targets, previews of events still ahead and sentiment are not reasons" in text
    assert 'Begin the Rationale with "What changed since the previous record:"' in text
    assert "A rating off Hold, kept or new, stands only when you can state" in text
    assert text.index(PRIOR) < text.index("**Debate History:**")


@pytest.mark.unit
def test_without_one_the_prompt_is_unchanged():
    text = _prompt()
    assert text == _prompt("") == _prompt("  \n")
    assert "Start from Hold: the market prices most stocks about right" in text
    assert "Move off Hold only when you can state" in text and "previous research record" not in text


@pytest.mark.unit
def test_the_previous_record_is_part_of_the_run_state_and_its_checkpoint_signature(tmp_path):
    graph = _bare_graph(tmp_path)
    assert graph.create_run_state("BKNG", "2026-11-03", prior_record=PRIOR)["prior_record"] == PRIOR
    assert graph.create_run_state("BKNG", "2026-11-03")["prior_record"] == ""
    assert graph._run_signature("stock") == graph._run_signature("stock", prior_record="")
    assert graph._run_signature("stock") != graph._run_signature("stock", prior_record=PRIOR)
