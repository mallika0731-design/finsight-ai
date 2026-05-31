"""
ui/insight_view.py
------------------
Automatic financial-ratio extraction and loan-eligibility scoring.

Extraction strategy:
  1. Regex keyword search with a character window around each keyword
  2. Indian number format support: ₹ 1,23,456 / "45 lakhs" / "3.2 crores"
  3. Manual override inputs so judges / users can correct any missed values

Loan score:
  Rule-based scorer (0–100) modelled on standard MSME bank-lending criteria.
  Four sub-dimensions: profitability, liquidity, leverage, efficiency.
  Each dimension is independently scoreable — if a value is missing, that
  dimension is simply excluded rather than zeroing the whole score.
"""

import re
import streamlit as st
import plotly.graph_objects as go


# ── Number parsing ────────────────────────────────────────────────────────────

def _parse_number(token: str) -> float | None:
    """
    Parse a numeric string that may use Indian formats.

    Handles:
      "1,23,456"   → 123456.0
      "45 lakhs"   → 4500000.0
      "3.2 crores" → 32000000.0
      "678.90"     → 678.9

    Returns None if no valid number is found.
    """
    token = token.strip()

    # Crore / CR
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:crore|crores|cr\.?)", token, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        return val * 1e7

    # Lakh / LAC
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|lac\.?)", token, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        return val * 1e5

    # Plain number (with possible commas)
    m = re.fullmatch(r"[\d,]+(?:\.\d+)?", token.replace(" ", ""))
    if m:
        try:
            return float(token.replace(",", ""))
        except ValueError:
            return None

    return None


def _find_near_keyword(full_text: str, keywords: list, window: int = 150) -> float | None:
    """
    Search *full_text* for the first numeric value appearing within *window*
    characters after any of the given keywords.

    Returns the first valid positive number found, or None.
    """
    text_lower = full_text.lower()

    for kw in keywords:
        pos = text_lower.find(kw.lower())
        while pos != -1:
            snippet = full_text[pos: pos + window]

            # Match candidate tokens: digits, commas, dots, unit suffixes
            candidates = re.findall(
                r"\b\d[\d,]*(?:\.\d+)?\s*(?:crore|crores|cr|lakh|lakhs|lac)?\b",
                snippet,
                re.I,
            )
            for cand in candidates:
                val = _parse_number(cand)
                if val is not None and val > 0:
                    return val

            pos = text_lower.find(kw.lower(), pos + 1)

    return None


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_financials(full_text: str) -> dict:
    """
    Extract six key financial figures from document text.

    Returns
    -------
    dict with keys: revenue, net_profit, total_assets, total_liabilities,
    current_assets, current_liabilities.  Values are floats (INR) or None.
    """
    return {
        "revenue": _find_near_keyword(full_text, [
            "total revenue", "revenue from operations", "net revenue",
            "turnover", "total income", "gross revenue", "total sales",
        ]),
        "net_profit": _find_near_keyword(full_text, [
            "net profit", "profit after tax", "pat",
            "profit for the year", "net income", "net earnings",
        ]),
        "total_assets": _find_near_keyword(full_text, [
            "total assets", "total asset",
        ]),
        "total_liabilities": _find_near_keyword(full_text, [
            "total liabilities", "total liability",
            "total debt", "total borrowings",
        ]),
        "current_assets": _find_near_keyword(full_text, [
            "current assets", "total current assets",
        ]),
        "current_liabilities": _find_near_keyword(full_text, [
            "current liabilities", "total current liabilities",
        ]),
    }


# ── Ratio computation ─────────────────────────────────────────────────────────

def compute_ratios(f: dict) -> dict:
    """
    Derive standard financial ratios from extracted figures.

    Each ratio is None if the required inputs are unavailable or would
    cause a division-by-zero.
    """
    rev   = f.get("revenue")
    np_   = f.get("net_profit")
    ta    = f.get("total_assets")
    tl    = f.get("total_liabilities")
    ca    = f.get("current_assets")
    cl    = f.get("current_liabilities")

    ratios: dict = {}

    # Net profit margin (%)
    ratios["net_profit_margin"] = (
        (np_ / rev * 100) if (rev and np_ is not None and rev > 0) else None
    )

    # Current ratio
    ratios["current_ratio"] = (
        (ca / cl) if (ca and cl and cl > 0) else None
    )

    # Debt-to-equity ratio  (D/E = Total Liabilities / Equity;  Equity = TA - TL)
    if ta and tl and tl > 0:
        equity = ta - tl
        ratios["debt_to_equity"] = (tl / equity) if equity > 0 else None
    else:
        ratios["debt_to_equity"] = None

    # Asset turnover
    ratios["asset_turnover"] = (
        (rev / ta) if (rev and ta and ta > 0) else None
    )

    return ratios


# ── Loan scoring ──────────────────────────────────────────────────────────────

def compute_loan_score(ratios: dict) -> dict:
    """
    Rule-based creditworthiness score (0–100) over four dimensions.

    Missing dimensions are excluded; the score is normalised over available
    points so a partially complete document still gets a fair rating.

    Returns
    -------
    dict with keys: score (int), grade (str), recommendation (str),
    breakdown (dict of dimension → (earned, max, display_value)).
    """
    score     = 0
    max_score = 0
    breakdown = {}

    # ── Dimension 1: Profitability — Net Profit Margin (max 30 pts) ──────────
    npm = ratios.get("net_profit_margin")
    if npm is not None:
        max_score += 30
        if   npm >= 15: pts = 30
        elif npm >= 10: pts = 22
        elif npm >=  5: pts = 14
        elif npm >=  0: pts = 6
        else:           pts = 0
        score += pts
        breakdown["Profitability"] = (pts, 30, f"{npm:.1f}%")

    # ── Dimension 2: Liquidity — Current Ratio (max 25 pts) ──────────────────
    cr = ratios.get("current_ratio")
    if cr is not None:
        max_score += 25
        if   cr >= 2.0: pts = 25
        elif cr >= 1.5: pts = 18
        elif cr >= 1.0: pts = 10
        else:           pts = 0
        score += pts
        breakdown["Liquidity"] = (pts, 25, f"{cr:.2f}x")

    # ── Dimension 3: Leverage — D/E Ratio (max 25 pts) ───────────────────────
    de = ratios.get("debt_to_equity")
    if de is not None:
        max_score += 25
        if   de <= 0.5: pts = 25
        elif de <= 1.0: pts = 18
        elif de <= 2.0: pts = 10
        elif de <= 3.0: pts = 4
        else:           pts = 0
        score += pts
        breakdown["Leverage"] = (pts, 25, f"{de:.2f}x")

    # ── Dimension 4: Efficiency — Asset Turnover (max 20 pts) ────────────────
    at_ = ratios.get("asset_turnover")
    if at_ is not None:
        max_score += 20
        if   at_ >= 1.5: pts = 20
        elif at_ >= 1.0: pts = 14
        elif at_ >= 0.5: pts = 8
        else:            pts = 3
        score += pts
        breakdown["Efficiency"] = (pts, 20, f"{at_:.2f}x")

    # ── Guard: not enough data ────────────────────────────────────────────────
    if max_score == 0:
        return {
            "score": 0,
            "grade": "N/A",
            "recommendation": "Insufficient financial data in the document.",
            "breakdown": {},
        }

    # Normalise to 100 if not all dimensions were available
    normalised = round((score / max_score) * 100)

    if   normalised >= 75: grade, rec = "A", "Strong candidate — eligible for most MSME loan products."
    elif normalised >= 55: grade, rec = "B", "Moderate eligibility — may qualify with collateral."
    elif normalised >= 35: grade, rec = "C", "Below average — improve profitability or reduce leverage first."
    else:                  grade, rec = "D", "High risk — significant improvement required before applying."

    return {
        "score": normalised,
        "grade": grade,
        "recommendation": rec,
        "breakdown": breakdown,
    }


# ── Plotly charts ─────────────────────────────────────────────────────────────

def _gauge(score: int, grade: str) -> go.Figure:
    needle_color = (
        "#22c55e" if score >= 75 else
        "#f59e0b" if score >= 55 else
        "#ef4444"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Creditworthiness — Grade {grade}", "font": {"color": "#e2e8f0", "size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
            "bar":  {"color": needle_color},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#2d0a0a"},
                {"range": [35, 55], "color": "#2d1b00"},
                {"range": [55, 75], "color": "#162400"},
                {"range": [75, 100],"color": "#0a2e1a"},
            ],
        },
        number={"font": {"color": "#e2e8f0", "size": 44}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        margin=dict(l=20, r=20, t=50, b=10),
        height=270,
    )
    return fig


def _breakdown_chart(breakdown: dict) -> go.Figure:
    dims    = list(breakdown.keys())
    earned  = [v[0] for v in breakdown.values()]
    maxpts  = [v[1] for v in breakdown.values()]
    labels  = [v[2] for v in breakdown.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Score",
        x=dims,
        y=earned,
        text=[f"{e}/{m}  ({lbl})" for e, m, lbl in zip(earned, maxpts, labels)],
        textposition="outside",
        marker_color="#818cf8",
    ))
    fig.add_trace(go.Bar(
        name="Remaining",
        x=dims,
        y=[m - e for e, m in zip(earned, maxpts)],
        marker_color="#1e293b",
        hovertemplate="Remaining: %{y}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font={"color": "#e2e8f0"},
        legend=dict(bgcolor="#0f1117", font=dict(color="#94a3b8")),
        xaxis=dict(tickfont=dict(size=11, color="#94a3b8")),
        yaxis=dict(title="Points", gridcolor="#1e293b", color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
    )
    return fig


# ── Main view ─────────────────────────────────────────────────────────────────

def run_insight_view() -> None:
    st.markdown("### 📊 Financial Insights & Loan Eligibility")

    if "full_text" not in st.session_state:
        st.warning("Please upload a document first (use the **Upload** page).")
        return

    full_text = st.session_state["full_text"]

    with st.spinner("Extracting financial figures…"):
        financials = extract_financials(full_text)
        ratios     = compute_ratios(financials)
        loan       = compute_loan_score(ratios)

    # ── Section 1: Extracted figures ─────────────────────────────────────────
    st.markdown("#### Extracted Financial Figures")
    _FIGURE_LABELS = [
        ("revenue",             "Revenue"),
        ("net_profit",          "Net Profit"),
        ("total_assets",        "Total Assets"),
        ("total_liabilities",   "Total Liabilities"),
        ("current_assets",      "Current Assets"),
        ("current_liabilities", "Current Liabilities"),
    ]

    cols = st.columns(3)
    for idx, (key, label) in enumerate(_FIGURE_LABELS):
        val = financials.get(key)
        display = f"₹ {val:,.0f}" if val is not None else "—"
        cols[idx % 3].metric(label=label, value=display)

    # ── Section 2: Computed ratios ────────────────────────────────────────────
    st.markdown("#### Computed Ratios")
    _RATIO_META = [
        ("net_profit_margin", "Net Profit Margin", "{:.1f}%"),
        ("current_ratio",     "Current Ratio",     "{:.2f}×"),
        ("debt_to_equity",    "Debt / Equity",     "{:.2f}×"),
        ("asset_turnover",    "Asset Turnover",    "{:.2f}×"),
    ]
    rcols = st.columns(4)
    for i, (key, label, fmt) in enumerate(_RATIO_META):
        val = ratios.get(key)
        rcols[i].metric(label=label, value=fmt.format(val) if val is not None else "—")

    st.markdown("---")

    # ── Section 3: Loan score + gauge ────────────────────────────────────────
    st.markdown("#### Loan Eligibility Score")

    if not loan["breakdown"]:
        st.warning(
            "Not enough financial data was found to compute a score.  \n"
            "Try using the manual override below, or upload a more detailed document."
        )
    else:
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(_gauge(loan["score"], loan["grade"]), use_container_width=True)
        with right:
            st.markdown(f"**Recommendation**")
            st.info(loan["recommendation"])
            st.plotly_chart(_breakdown_chart(loan["breakdown"]), use_container_width=True)

    # ── Section 4: Manual override ────────────────────────────────────────────
    with st.expander("✏️ Override extracted values (if the auto-extraction missed anything)"):
        st.caption(
            "Enter values in Indian Rupees (₹). Leave at 0 if not available."
        )
        c1, c2 = st.columns(2)
        ov = {
            "revenue":             c1.number_input("Revenue (₹)",             value=float(financials.get("revenue")             or 0), min_value=0.0, step=100_000.0, format="%.0f"),
            "net_profit":          c2.number_input("Net Profit (₹)",          value=float(financials.get("net_profit")          or 0), step=10_000.0,  format="%.0f"),
            "total_assets":        c1.number_input("Total Assets (₹)",        value=float(financials.get("total_assets")        or 0), min_value=0.0, step=100_000.0, format="%.0f"),
            "total_liabilities":   c2.number_input("Total Liabilities (₹)",   value=float(financials.get("total_liabilities")   or 0), min_value=0.0, step=100_000.0, format="%.0f"),
            "current_assets":      c1.number_input("Current Assets (₹)",      value=float(financials.get("current_assets")      or 0), min_value=0.0, step=100_000.0, format="%.0f"),
            "current_liabilities": c2.number_input("Current Liabilities (₹)", value=float(financials.get("current_liabilities") or 0), min_value=0.0, step=100_000.0, format="%.0f"),
        }

        if st.button("Recalculate with these values", type="secondary"):
            # Replace 0 with None so compute_ratios treats it as missing
            manual_fin    = {k: (v if v != 0.0 else None) for k, v in ov.items()}
            manual_ratios = compute_ratios(manual_fin)
            manual_loan   = compute_loan_score(manual_ratios)

            st.markdown("**Updated result:**")
            mc1, mc2 = st.columns(2)
            mc1.metric("Score", f"{manual_loan['score']} / 100")
            mc2.metric("Grade", manual_loan["grade"])
            st.info(manual_loan["recommendation"])
            if manual_loan["breakdown"]:
                st.plotly_chart(
                    _breakdown_chart(manual_loan["breakdown"]),
                    use_container_width=True,
                )
