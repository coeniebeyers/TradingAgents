"""The caller's book, as the decision agents see it.

Optional input to a run: what is held, at what average price, and how much cash
is free. Without it the agents cannot tell adding to a full position from
opening a new one. Three states are distinct and must stay so: a position, a
flat book, and no context at all, since treating "not provided" as "flat" would
invent a fact about the caller's account.

Broker-neutral by construction: quantities are generic units and the currency is
whatever label the caller passes, so nothing here implies a venue or an
execution path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

SIZING_HEADER = "Sizing (computed by code from the ratings and the book):"


class Position(BaseModel):
    ticker: str = Field(description="Instrument symbol, e.g. AAPL")
    quantity: float = Field(description="Signed units held; negative is short")
    average_price: float | None = Field(default=None, description="Average entry price per unit")
    market_value: float | None = Field(default=None, description="Current value of the position, in the book's currency")


class PortfolioContext(BaseModel):
    cash: float | None = Field(default=None, description="Free cash available")
    currency: str | None = Field(default=None, description="Currency label for cash and prices")
    positions: list[Position] = Field(default_factory=list)
    sizing: str | None = Field(default=None, description="The caller's sizing brief for the analyzed instrument: its "
                               "target weight and how the book can fund a change, computed by code")

    def position_in(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker.upper() == ticker.strip().upper()), None)

    def total_value(self) -> float | None:
        """Positions plus cash, when every position has a value; else None."""
        if not self.positions or any(p.market_value is None for p in self.positions):
            return None
        return sum(p.market_value for p in self.positions) + (self.cash or 0.0)

    def render(self, ticker: str) -> str:
        """The portfolio block for the decision agents, led by the analyzed instrument."""
        symbol = ticker.strip().upper()
        held = self.position_in(symbol)
        total = self.total_value()

        def worth(p: Position) -> str:
            return f", worth {p.market_value:,.0f} ({p.market_value / total:.1%} of the book)" if total else ""

        if held is None:
            lines = [f"- No current position in {symbol}"]
        else:
            price = f", average price {held.average_price:,.2f}" if held.average_price is not None else ""
            lines = [f"- Current position in {symbol}: {held.quantity:,.4g} units{price}{worth(held)}"]
        if self.cash is not None:
            lines.append(f"- Cash available: {self.cash:,.2f}{' ' + self.currency if self.currency else ''}")
        if total:
            lines.append(f"- Book value, positions and cash: {total:,.0f}")
        others = [p for p in self.positions if p is not held]
        if others:
            lines.append("- Other positions: " + ", ".join(
                f"{p.ticker.upper()} {p.quantity:,.4g}" + (f" ({p.market_value / total:.1%})" if total else "")
                for p in others))
        if self.sizing:
            lines.append(f"\n{SIZING_HEADER}\n{self.sizing.strip()}")
        return "Portfolio at the analysis date:\n" + "\n".join(lines)

    def fingerprint(self) -> str:
        """Stable digest of the book, so a changed one cannot resume a stale run."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def load_portfolio(path: str | Path) -> PortfolioContext:
    """Read a portfolio JSON file, failing before the run rather than mid-graph."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PortfolioContext.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"portfolio file {path} is not usable: {exc}") from exc
