"""
EDpyFlow — Heat Demand Scenario Analysis Dashboard

Loads a trained EDSurrogate model and lets users compare refurbishment
upgrade scenarios for residential building stocks.

Run:
    streamlit run dashboard/app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ── Constants ─────────────────────────────────────────────────────────────────

UPGRADE_MAP = {
    "standard → retrofit":     ("standard",  "retrofit"),
    "standard → adv_retrofit": ("standard",  "adv_retrofit"),
    "retrofit → adv_retrofit": ("retrofit",  "adv_retrofit"),
}
UPGRADE_PATHS = list(UPGRADE_MAP.keys())

SCENARIO_A_COLOR = "#1FA386"  # green
SCENARIO_B_COLOR = "#6E22B8"  # purple
BASELINE_COLOR   = "#94a3b8"  # slate

BUILDING_TYPE_LABELS = {
    "SFH": "[SFH] Single-Family House",
    "TH":  "[TH] Terraced House",
    "MFH": "[MFH] Multi-Family House",
    "AB":  "[AB] Apartment Block",
}

DISPLAY_COLS = [
    "building_type", "location", "refurbishment_status",
    "construction_year", "net_leased_area", "num_floors", "floor_height",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EDpyFlow — Scenario Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.html("<style>section[data-testid='stSidebar'], .main { zoom: 0.9; } .block-container { padding-top: 3.5rem; }</style>")
st.title("EDpyFlow — Heat Demand Scenario Analysis")
st.caption(
    "Compare refurbishment upgrade scenarios for residential building stocks, "
    "powered by the EDpyFlow pipeline."
)

# ── Sidebar: run + model + building stock ─────────────────────────────────────

st.sidebar.header("Configuration")

runs_dir = "runs"
if not os.path.isdir(runs_dir):
    st.error(
        "No `runs/` directory found. Run the dashboard from the `EDpyFlow/` directory:\n"
        "```\nstreamlit run dashboard/app.py\n```"
    )
    st.stop()

available_runs = sorted([
    d for d in os.listdir(runs_dir)
    if os.path.isdir(os.path.join(runs_dir, d, "models"))
])
if not available_runs:
    st.error("No runs with a trained model found in `runs/`.")
    st.stop()

selected_run = st.sidebar.selectbox("Run", available_runs)
run_dir      = os.path.join(runs_dir, selected_run)

available_models = sorted([
    f for f in os.listdir(os.path.join(run_dir, "models"))
    if f.endswith(".json")
])
if not available_models:
    st.error(f"No model (.json) files found in `{run_dir}/models/`.")
    st.stop()
selected_model = st.sidebar.selectbox(
    "Model", available_models,
    format_func=lambda f: os.path.splitext(f)[0],
)
model_path = os.path.join(run_dir, "models", selected_model)


@st.cache_resource
def load_model(path: str) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m


model         = load_model(model_path)
feature_names = model.get_booster().feature_names

stock_source = st.sidebar.radio(
    "Building stock",
    ["Test set (pre-loaded)", "Upload CSV"],
)


@st.cache_data
def load_test_set(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "synthetic_dataset", "test_set.csv")
    if not os.path.exists(path):
        st.error(f"`test_set.csv` not found in `{run_dir}/synthetic_dataset/`. Re-run `train_surrogate.py` to generate it.")
        st.stop()
    return pd.read_csv(path, index_col="id")


if stock_source == "Test set (pre-loaded)":
    stock_df = load_test_set(run_dir)
    stock_df = stock_df.drop(columns=["total_energy"], errors="ignore")
else:
    uploaded = st.sidebar.file_uploader(
        "CSV file", type="csv",
        help=(
            "Required columns: construction_year, net_leased_area, num_floors, "
            "floor_height, building_type, location, refurbishment_status"
        ),
    )
    if uploaded is None:
        st.info(
            "Upload a CSV to continue.\n\n"
            "**Required columns:** `construction_year`, `net_leased_area`, "
            "`num_floors`, `floor_height`, `building_type`, `location`, "
            "`refurbishment_status`\n\n"
            "Optional: `id` column as index."
        )
        st.stop()
    stock_df = pd.read_csv(uploaded)
    if "id" in stock_df.columns:
        stock_df = stock_df.set_index("id")
    stock_df = stock_df.drop(columns=["total_energy"], errors="ignore")

st.sidebar.markdown(f"**Stock:** {len(stock_df)} buildings")
st.sidebar.markdown(f"**Locations:** {', '.join(l.capitalize() for l in sorted(stock_df['location'].unique()))}")
st.sidebar.markdown(f"**Building types:** {', '.join(BUILDING_TYPE_LABELS.get(t, t) for t in sorted(stock_df['building_type'].unique()))}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_energy(df: pd.DataFrame, target_status: str = None) -> np.ndarray:
    """Predict annual heat demand [kWh]. If target_status is None, uses each building's
    own refurbishment_status; otherwise overrides all buildings to target_status."""
    d = df.copy()
    if target_status is not None:
        d["refurbishment_status"] = target_status
    encoded = pd.get_dummies(d, columns=["building_type", "location", "refurbishment_status"])
    for col in feature_names:
        if col not in encoded.columns:
            encoded[col] = 0
    return model.predict(encoded[feature_names])


def run_scenario(
    filtered_df: pd.DataFrame,
    from_level: str,
    to_level: str,
    coverage_type: str,
    coverage_value,
    rank_by: str = "Savings",
) -> pd.DataFrame:
    """
    Applies an upgrade scenario to filtered_df.

    Returns filtered_df with added columns:
        current_energy   [kWh]
        scenario_energy  [kWh]
        savings          [kWh]
        savings_per_m2   [kWh/m²]
        upgraded         bool
    """
    df = filtered_df.copy()

    df["current_energy"]  = predict_energy(df)
    df["scenario_energy"] = df["current_energy"].copy()
    df["upgraded"]        = False

    eligible_mask = df["refurbishment_status"] == from_level
    if not eligible_mask.any():
        df["savings"] = 0.0
        return df

    eligible = df[eligible_mask].copy()
    upgraded_energy             = predict_energy(eligible, to_level)
    eligible["upgraded_energy"]   = upgraded_energy
    eligible["potential_savings"] = eligible["current_energy"] - upgraded_energy
    eligible["potential_savings_per_m2"] = (
        eligible["potential_savings"] / eligible["net_leased_area"]
    )

    rank_col = "potential_savings_per_m2" if rank_by == "Savings per m²" else "potential_savings"

    if coverage_type == "all":
        selected_idx = eligible.index
    elif coverage_type == "top_pct":
        n = max(1, int(len(eligible) * coverage_value / 100))
        selected_idx = eligible.nlargest(n, rank_col).index
    else:  # top_n
        n = min(int(coverage_value), len(eligible))
        selected_idx = eligible.nlargest(n, rank_col).index

    df.loc[selected_idx, "scenario_energy"] = eligible.loc[selected_idx, "upgraded_energy"]
    df.loc[selected_idx, "upgraded"]        = True
    df["savings"] = df["current_energy"] - df["scenario_energy"]
    df["savings_per_m2"] = df["savings"] / df["net_leased_area"]
    return df


def fig_by_decade(result: pd.DataFrame, divisor: float, unit: str,
                  traces: list[tuple[str, str, str]]) -> go.Figure:
    """
    Bar chart of baseline vs after-upgrade demand grouped by construction decade.
    traces: list of (label, energy_col, color) — energy_col is 'current_energy' or 'scenario_energy'
    """
    result = result.copy()
    result["decade"] = (result["construction_year"] // 10 * 10).astype(str) + "s"
    decades = sorted(result["decade"].unique())
    fig = go.Figure()
    for label, col, color in traces:
        vals = [result[result["decade"] == d][col].sum() / divisor for d in decades]
        fig.add_trace(go.Bar(name=label, x=decades, y=vals, marker_color=color))
    fig.update_layout(barmode="group", yaxis_title=f"Energy demand [{unit}]", height=380,
                      xaxis_title="Construction decade",
                      legend=dict(orientation="h", y=1.08))
    return fig


def fig_cumulative(results: list[tuple[str, pd.DataFrame, str, str]],
                   divisor: float, unit: str) -> go.Figure:
    """
    Cumulative savings curve.
    results: list of (label, result_df, color, rank_by)
        label:      legend label for the curve
        result_df:  output of run_scenario()
        color:      line color
        rank_by:    'Savings' or 'Savings per m²' — priority order for adding buildings
    X axis: number of buildings upgraded in priority order.
    Y axis: cumulative absolute savings [unit].
    """
    fig = go.Figure()
    for label, result, color, rank_by in results:
        sort_col = "savings_per_m2" if rank_by == "Savings per m²" else "savings"
        upgraded = result[result["upgraded"]].sort_values(sort_col, ascending=False)
        if upgraded.empty:
            continue
        cumulative = upgraded["savings"].cumsum() / divisor
        fig.add_trace(go.Scatter(
            name=label, x=list(range(1, len(cumulative) + 1)), y=cumulative,
            mode="lines", line=dict(color=color, width=2),
        ))
    fig.update_layout(
        xaxis_title="Number of buildings upgraded",
        yaxis_title=f"Cumulative savings [{unit}]",
        height=400,
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def smart_unit(total_kwh: float) -> tuple[str, float]:
    if total_kwh >= 1e9:
        return "GWh", 1e6
    if total_kwh >= 1e6:
        return "MWh", 1e3
    return "kWh", 1.0


def scenario_controls(label: str, default_path: str, key: str) -> tuple:
    """
    Renders scenario filter + upgrade config UI.
    Returns (filtered_df, from_level, to_level, coverage_type, coverage_value, rank_by).
    """
    if label:
        st.subheader(f"Scenario {label}")

    all_locations = sorted(stock_df["location"].unique())
    all_types     = sorted(stock_df["building_type"].unique())

    st.markdown("**Filter building stock**")
    with st.container(border=True):
        sel_loc = st.multiselect(
            "Location", all_locations, default=all_locations,
            format_func=lambda l: l.capitalize(),
            key=f"{key}_loc",
        )
        sel_type = st.multiselect(
            "Building type", all_types, default=all_types,
            format_func=lambda t: BUILDING_TYPE_LABELS.get(t, t),
            key=f"{key}_type",
        )
        # Compute area range from current loc/type selection
        _tmp = stock_df.copy()
        if sel_loc:
            _tmp = _tmp[_tmp["location"].isin(sel_loc)]
        if sel_type:
            _tmp = _tmp[_tmp["building_type"].isin(sel_type)]
        area_min = float(_tmp["net_leased_area"].min()) if len(_tmp) > 0 else 0.0
        area_max = float(_tmp["net_leased_area"].max()) if len(_tmp) > 0 else 1.0
        if area_min == area_max:
            area_max = area_min + 1.0

        # Reset slider whenever the range changes (e.g. different location selected)
        _range_key = f"{key}_area_range"
        _area_key  = f"{key}_area"
        if st.session_state.get(_range_key) != (area_min, area_max):
            st.session_state[_range_key] = (area_min, area_max)
            if _area_key in st.session_state:
                st.session_state[_area_key] = (area_min, area_max)

        sel_area = st.slider(
            "Net leased area (m²)", min_value=area_min, max_value=area_max,
            value=(area_min, area_max), step=1.0,
            key=f"{key}_area",
        )

    path = st.selectbox(
        "Upgrade path", UPGRADE_PATHS,
        index=UPGRADE_PATHS.index(default_path),
        key=f"{key}_path",
    )
    from_level, to_level = UPGRADE_MAP[path]

    col_cov, _, col_rank = st.columns([1.2, 0.15, 1])

    with col_cov:
        cov_label = st.radio(
            "Coverage",
            ["All eligible buildings", "Top %", "Top N"],
            key=f"{key}_cov_type",
            horizontal=True,
        )
        coverage_type  = "all"
        coverage_value = None
        if cov_label == "Top %":
            coverage_type  = "top_pct"
            coverage_value = st.slider("% of eligible buildings to upgrade", 1, 100, 30, key=f"{key}_pct")
        elif cov_label == "Top N":
            coverage_type  = "top_n"
            coverage_value = st.number_input(
                "Number of buildings to upgrade", min_value=1, value=20, step=5, key=f"{key}_n"
            )

    with col_rank:
        rank_by = st.radio(
            "Rank by",
            ["Savings", "Savings per m²"],
            key=f"{key}_rank",
            horizontal=True,
        )

    filtered = stock_df.copy()
    if sel_loc:
        filtered = filtered[filtered["location"].isin(sel_loc)]
    if sel_type:
        filtered = filtered[filtered["building_type"].isin(sel_type)]
    filtered = filtered[
        filtered["net_leased_area"].between(sel_area[0], sel_area[1])
    ]

    n_eligible = (filtered["refurbishment_status"] == from_level).sum()
    st.caption(
        f"{len(filtered)} buildings in filtered stock · "
        f"{n_eligible} eligible for upgrade ({from_level})"
    )

    return filtered, from_level, to_level, coverage_type, coverage_value, rank_by


def format_result_table(
    result: pd.DataFrame, unit: str, divisor: float, rank_by: str = "Savings"
) -> pd.DataFrame:
    """Format result DataFrame for display, converting energy columns to the given unit."""
    cols = DISPLAY_COLS + ["current_energy", "scenario_energy", "savings"]
    if rank_by == "Savings per m²":
        cols = cols + ["savings_per_m2"]
    t = result[cols].copy()
    t["current_energy"]  = (t["current_energy"]  / divisor).round(2)
    t["scenario_energy"] = (t["scenario_energy"] / divisor).round(2)
    t["savings"]         = (t["savings"]         / divisor).round(2)
    rename = {
        "current_energy":  f"current [{unit}]",
        "scenario_energy": f"after upgrade [{unit}]",
        "savings":         f"savings [{unit}]",
    }
    if rank_by == "Savings per m²":
        t["savings_per_m2"] = (t["savings_per_m2"] / divisor).round(4)
        rename["savings_per_m2"] = f"savings/m² [{unit}/m²]"
    return t.rename(columns=rename)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_single, tab_compare = st.tabs(["Single Scenario", "Scenario Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single Scenario
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:
    params_s = scenario_controls("", "standard → retrofit", "s")

    run_single = st.button("Run Analysis", type="primary", use_container_width=True, key="run_single")

    if run_single:
        with st.spinner("Running…"):
            result_s = run_scenario(*params_s)

        unit, divisor = smart_unit(result_s["current_energy"].sum())

        def fmt_s(kwh: float) -> str:
            return f"{kwh / divisor:,.1f} {unit}"

        base_s = result_s["current_energy"].sum()
        sav_s  = result_s["savings"].sum()
        upg_s  = result_s["upgraded"].sum()

        # ── Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total baseline demand",    fmt_s(base_s))
        m2.metric("Demand after upgrade",     fmt_s(base_s - sav_s),
                  delta=f"-{fmt_s(sav_s)}", delta_color="inverse")
        m3.metric("Savings",                  f"{sav_s / base_s * 100:.1f}%")
        m4.metric("Buildings upgraded",       int(upg_s))

        # ── Savings by city ──
        st.subheader("Energy demand by city")
        cities    = sorted(result_s["location"].unique())
        city_base = [result_s[result_s["location"] == c]["current_energy"].sum() / divisor for c in cities]
        city_post = [result_s[result_s["location"] == c]["scenario_energy"].sum() / divisor for c in cities]

        fig_city_s = go.Figure()
        fig_city_s.add_trace(go.Bar(name="Baseline",      x=cities, y=city_base, marker_color=BASELINE_COLOR))
        fig_city_s.add_trace(go.Bar(name="After upgrade", x=cities, y=city_post, marker_color=SCENARIO_A_COLOR))
        fig_city_s.update_layout(barmode="group", yaxis_title=f"Energy demand [{unit}]", height=370,
                                 legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_city_s, use_container_width=True)

        # ── Savings by building type ──
        st.subheader("Energy demand by building type")
        btypes     = sorted(result_s["building_type"].unique())
        type_base  = [result_s[result_s["building_type"] == t]["current_energy"].sum() / divisor for t in btypes]
        type_post  = [result_s[result_s["building_type"] == t]["scenario_energy"].sum() / divisor for t in btypes]

        fig_type_s = go.Figure()
        fig_type_s.add_trace(go.Bar(name="Baseline",      x=btypes, y=type_base, marker_color=BASELINE_COLOR))
        fig_type_s.add_trace(go.Bar(name="After upgrade", x=btypes, y=type_post, marker_color=SCENARIO_A_COLOR))
        fig_type_s.update_layout(barmode="group", yaxis_title=f"Energy demand [{unit}]", height=370,
                                 legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_type_s, use_container_width=True)

        # ── Savings by construction decade ──
        st.subheader("Energy demand by construction decade")
        st.plotly_chart(fig_by_decade(result_s, divisor, unit, [
            ("Baseline",      "current_energy",  BASELINE_COLOR),
            ("After upgrade", "scenario_energy", SCENARIO_A_COLOR),
        ]), use_container_width=True)

        # ── Cumulative savings curve ──
        st.subheader("Cumulative savings")
        _rank_by_s  = params_s[5]
        _rank_label_s = "savings/m²" if _rank_by_s == "Savings per m²" else "savings"
        st.caption(f"Buildings ranked by {_rank_label_s} (highest first). Shows how much of the total potential is captured as more buildings are upgraded.")
        st.plotly_chart(fig_cumulative(
            [("Scenario", result_s, SCENARIO_A_COLOR, _rank_by_s)], divisor, unit
        ), use_container_width=True)

        # ── Eligible buildings table ──
        st.subheader("Upgraded buildings")
        _table_sort_label_s = "savings/m²" if params_s[5] == "Savings per m²" else "savings"
        st.caption(f"All upgraded buildings, sorted by {_table_sort_label_s}.")

        _sort_col_s = "savings_per_m2" if params_s[5] == "Savings per m²" else "savings"
        eligible_result = result_s[result_s["upgraded"]].sort_values(_sort_col_s, ascending=False)
        if eligible_result.empty:
            st.info("No buildings were upgraded in this scenario.")
        else:
            display = format_result_table(eligible_result, unit, divisor, params_s[5])
            st.dataframe(display, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Scenario Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        params_a = scenario_controls("A", "standard → retrofit",     "a")
    with col_b:
        params_b = scenario_controls("B", "standard → adv_retrofit", "b")

    run_compare = st.button("Run Comparison", type="primary", use_container_width=True, key="run_compare")

    if run_compare:
        with st.spinner("Running Scenario A…"):
            result_a = run_scenario(*params_a)
        with st.spinner("Running Scenario B…"):
            result_b = run_scenario(*params_b)

        peak = max(result_a["current_energy"].sum(), result_b["current_energy"].sum())
        unit, divisor = smart_unit(peak)

        def fmt_c(kwh: float) -> str:
            return f"{kwh / divisor:,.1f} {unit}"

        base_a = result_a["current_energy"].sum()
        base_b = result_b["current_energy"].sum()
        sav_a  = result_a["savings"].sum()
        sav_b  = result_b["savings"].sum()
        upg_a  = result_a["upgraded"].sum()
        upg_b  = result_b["upgraded"].sum()

        # ── Metrics ──
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Baseline A", fmt_c(base_a))
            st.metric("Baseline B", fmt_c(base_b))
        with m2:
            st.metric("Scenario A demand", fmt_c(base_a - sav_a),
                      delta=f"-{fmt_c(sav_a)}", delta_color="inverse")
            st.metric("Scenario B demand", fmt_c(base_b - sav_b),
                      delta=f"-{fmt_c(sav_b)}", delta_color="inverse")
        with m3:
            st.metric("A savings", f"{sav_a / base_a * 100:.1f}%")
            st.metric("B savings", f"{sav_b / base_b * 100:.1f}%")
        with m4:
            st.metric("A buildings upgraded", int(upg_a))
            st.metric("B buildings upgraded", int(upg_b))

        # ── Total demand bar chart ──
        st.subheader("Total annual heat demand")
        fig_total = go.Figure()
        fig_total.add_trace(go.Bar(
            name="Baseline", x=["Scenario A", "Scenario B"],
            y=[base_a / divisor, base_b / divisor],
            marker_color=BASELINE_COLOR, offsetgroup=0,
        ))
        fig_total.add_trace(go.Bar(
            name="After upgrade A", x=["Scenario A"],
            y=[(base_a - sav_a) / divisor],
            marker_color=SCENARIO_A_COLOR, offsetgroup=1,
        ))
        fig_total.add_trace(go.Bar(
            name="After upgrade B", x=["Scenario B"],
            y=[(base_b - sav_b) / divisor],
            marker_color=SCENARIO_B_COLOR, offsetgroup=1,
        ))
        fig_total.update_layout(barmode="group", yaxis_title=f"Energy demand [{unit}]", height=380,
                                legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_total, use_container_width=True)

        # ── Cumulative savings curve ──
        st.subheader("Cumulative savings")
        _rank_by_a    = params_a[5]
        _rank_by_b    = params_b[5]
        _rank_label_a = "savings/m²" if _rank_by_a == "Savings per m²" else "savings"
        _rank_label_b = "savings/m²" if _rank_by_b == "Savings per m²" else "savings"
        _rank_note = (
            f"A ranked by {_rank_label_a}, B ranked by {_rank_label_b}."
            if _rank_label_a != _rank_label_b
            else f"Both ranked by {_rank_label_a} (highest first)."
        )
        st.caption(f"{_rank_note} Shows how much of the total potential is captured as more buildings are upgraded.")
        st.plotly_chart(fig_cumulative([
            ("Scenario A", result_a, SCENARIO_A_COLOR, _rank_by_a),
            ("Scenario B", result_b, SCENARIO_B_COLOR, _rank_by_b),
        ], divisor, unit), use_container_width=True)

        # ── Buildings table ──
        st.subheader("Upgraded buildings")

        res_tab_a, res_tab_b = st.tabs(["Scenario A", "Scenario B"])
        for res_tab, result, params in [(res_tab_a, result_a, params_a), (res_tab_b, result_b, params_b)]:
            with res_tab:
                _sort_col_c = "savings_per_m2" if params[5] == "Savings per m²" else "savings"
                _sort_label_c = "savings/m²" if params[5] == "Savings per m²" else "savings"
                st.caption(f"All upgraded buildings, sorted by {_sort_label_c}.")
                upgraded = result[result["savings"] > 0].sort_values(_sort_col_c, ascending=False)
                if upgraded.empty:
                    st.info("No buildings upgraded in this scenario.")
                else:
                    st.dataframe(format_result_table(upgraded, unit, divisor, params[5]), use_container_width=True)
