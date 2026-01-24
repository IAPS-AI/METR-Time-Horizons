"""
METR Benchmark Results Dashboard

An interactive dashboard that displays AI model capability horizons over time,
pulling data dynamically from https://metr.org/assets/benchmark_results.yaml
"""

import requests
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional


# Page configuration
st.set_page_config(
    page_title="METR Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_benchmark_data() -> dict:
    """Fetch benchmark results from METR's YAML endpoint."""
    url = "https://metr.org/assets/benchmark_results.yaml"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return yaml.safe_load(response.text)


def format_model_name(model_id: str) -> str:
    """Convert model ID to display name."""
    name_map = {
        "claude_3_5_sonnet": "Claude 3.5 Sonnet",
        "claude_3_5_sonnet_20241022": "Claude 3.5 Sonnet (Oct 2024)",
        "claude_3_7_sonnet": "Claude 3.7 Sonnet",
        "claude_3_opus": "Claude 3 Opus",
        "claude_4_opus": "Claude 4 Opus",
        "claude_4_1_opus": "Claude 4.1 Opus",
        "claude_4_sonnet": "Claude 4 Sonnet",
        "claude_sonnet_4_5": "Claude Sonnet 4.5",
        "claude_opus_4_5": "Claude Opus 4.5",
        "gpt2": "GPT-2",
        "gpt_3_5_turbo_instruct": "GPT-3.5 Turbo Instruct",
        "gpt_4": "GPT-4",
        "gpt_4_0125": "GPT-4 (Jan 2024)",
        "gpt_4_1106": "GPT-4 Turbo (Nov 2023)",
        "gpt_4_turbo": "GPT-4 Turbo",
        "gpt_4o": "GPT-4o",
        "gpt_5": "GPT-5",
        "gpt_5_1_codex_max": "GPT-5.1 Codex Max",
        "gpt-oss-120b": "GPT-OSS-120B",
        "o1_preview": "o1 Preview",
        "o1_elicited": "o1 (Elicited)",
        "o3": "o3",
        "o4-mini": "o4-mini",
        "davinci_002": "Davinci-002",
        "deepseek_r1": "DeepSeek R1",
        "deepseek_r1_0528": "DeepSeek R1 (May 2025)",
        "deepseek_v3": "DeepSeek V3",
        "deepseek_v3_0324": "DeepSeek V3 (Mar 2025)",
        "gemini_2_5_pro_preview": "Gemini 2.5 Pro Preview",
        "grok_4": "Grok 4",
        "kimi_k2_thinking": "Kimi K2 Thinking",
        "qwen_2_5_72b": "Qwen 2.5 72B",
        "qwen_2_72b": "Qwen 2 72B",
    }
    return name_map.get(model_id, model_id.replace("_", " ").title())


def get_model_family(model_id: str) -> str:
    """Categorize model into family for coloring."""
    if "claude" in model_id.lower():
        return "Anthropic"
    elif "gpt" in model_id.lower() or model_id.startswith("o1") or model_id.startswith("o3") or model_id.startswith("o4"):
        return "OpenAI"
    elif "deepseek" in model_id.lower():
        return "DeepSeek"
    elif "gemini" in model_id.lower():
        return "Google"
    elif "qwen" in model_id.lower():
        return "Alibaba"
    elif "grok" in model_id.lower():
        return "xAI"
    elif "kimi" in model_id.lower():
        return "Moonshot"
    else:
        return "Other"


def parse_results_to_dataframe(data: dict) -> pd.DataFrame:
    """Parse YAML results into a pandas DataFrame."""
    rows = []
    for model_id, model_data in data.get("results", {}).items():
        metrics = model_data.get("metrics", {})
        p50 = metrics.get("p50_horizon_length", {})
        p80 = metrics.get("p80_horizon_length", {})
        avg_score = metrics.get("average_score", {})
        usage = metrics.get("usage", {})

        # Convert minutes to hours for all horizon values
        rows.append({
            "model_id": model_id,
            "model_name": format_model_name(model_id),
            "family": get_model_family(model_id),
            "release_date": pd.to_datetime(model_data.get("release_date")),
            "p50_estimate": p50.get("estimate") / 60 if p50.get("estimate") else None,  # Convert to hours
            "p50_ci_low": p50.get("ci_low") / 60 if p50.get("ci_low") else None,
            "p50_ci_high": p50.get("ci_high") / 60 if p50.get("ci_high") else None,
            "p80_estimate": p80.get("estimate") / 60 if p80.get("estimate") else None,
            "p80_ci_low": p80.get("ci_low") / 60 if p80.get("ci_low") else None,
            "p80_ci_high": p80.get("ci_high") / 60 if p80.get("ci_high") else None,
            "average_score": avg_score.get("estimate"),
            "is_sota": metrics.get("is_sota", False),
            "cost_usd": usage.get("usd"),
            "working_time": usage.get("working_time"),
            "scaffolds": ", ".join(str(s) for s in model_data.get("scaffolds", []) if s),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["release_date", "p50_estimate"])
    df = df.sort_values("release_date")
    return df


def format_time_label(hours: float) -> str:
    """Convert hours to human-readable time label."""
    if hours < 1/60:  # Less than 1 minute
        return f"{hours * 3600:.0f} sec"
    elif hours < 1:  # Less than 1 hour
        return f"{hours * 60:.0f} min"
    elif hours < 24:
        return f"{hours:.1f} hrs" if hours != int(hours) else f"{int(hours)} hrs"
    else:
        days = hours / 24
        return f"{days:.1f} days" if days != int(days) else f"{int(days)} days"


def format_hover_time(hours: float) -> str:
    """Format time for hover - show hours as primary unit."""
    if hours < 1/60:  # Less than 1 minute
        return f"{hours * 3600:.1f} sec ({hours:.4f} hrs)"
    elif hours < 1:  # Less than 1 hour
        return f"{hours * 60:.1f} min ({hours:.2f} hrs)"
    elif hours < 24:
        return f"{hours:.2f} hrs"
    else:
        days = hours / 24
        return f"{hours:.1f} hrs ({days:.1f} days)"


def fit_exponential_trendline(
    df: pd.DataFrame,
    horizon_col: str = "p50_estimate",
    after_date: str | None = None
) -> Tuple[LinearRegression, float, pd.DataFrame]:
    """Fit an exponential trendline to the data."""
    fit_df = df.copy()
    if after_date:
        fit_df = fit_df[fit_df["release_date"] >= pd.Timestamp(after_date)]

    # Convert dates to numeric (days since epoch)
    X = fit_df["release_date"].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = np.log(fit_df[horizon_col].clip(lower=1e-6).values)

    reg = LinearRegression().fit(X, y)
    r_squared = reg.score(X, y)

    return reg, r_squared, fit_df


def generate_custom_projection(
    start_date: pd.Timestamp,
    start_value: float,
    doubling_time_days: float,
    end_date: pd.Timestamp,
    num_points: int = 50
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Generate a projection based on custom doubling time."""
    date_range = pd.date_range(start_date, end_date, periods=num_points)
    days_elapsed = np.array([(d - start_date).days for d in date_range])
    # Exponential growth: value = start_value * 2^(days/doubling_time)
    values = start_value * np.power(2, days_elapsed / doubling_time_days)
    return date_range, values


def calculate_time_to_horizon(
    current_value: float,
    target_value: float,
    doubling_time_days: float,
    from_date: pd.Timestamp
) -> pd.Timestamp:
    """Calculate when a target horizon will be reached given doubling time."""
    # target = current * 2^(days/doubling)
    # days = doubling * log2(target/current)
    days_needed = doubling_time_days * np.log2(target_value / current_value)
    return from_date + pd.Timedelta(days=days_needed)


def create_horizon_plot(
    df: pd.DataFrame,
    horizon_type: str = "p50",
    show_trendline: bool = True,
    show_ci: bool = True,
    selected_families: list = None,
    start_date: str = "2019-01-01",
    end_date: str = "2028-01-01",
    show_projection: bool = True,
    fit_start_date: str = "2020-01-01",
    custom_doubling_times: list = None,
    marker_size: int = 12,
) -> Tuple[go.Figure, Optional[float], Optional[float]]:
    """Create the interactive time horizon plot.

    Returns the figure and the fitted doubling time (if trendline shown).
    """

    # Color palette for model families
    family_colors = {
        "Anthropic": "#d97757",
        "OpenAI": "#18a683",
        "DeepSeek": "#5B8FF9",
        "Google": "#F6BD16",
        "Alibaba": "#E8684A",
        "xAI": "#9C5EDA",
        "Moonshot": "#FF9D4D",
        "Other": "#999999",
    }

    # Family markers
    family_markers = {
        "Anthropic": "diamond",
        "OpenAI": "circle",
        "DeepSeek": "square",
        "Google": "triangle-up",
        "Alibaba": "pentagon",
        "xAI": "star",
        "Moonshot": "hexagon",
        "Other": "x",
    }

    # Custom projection colors
    custom_colors = ["#e11d48", "#7c3aed", "#059669", "#d97706", "#0891b2"]

    # Filter by selected families
    if selected_families:
        df = df[df["family"].isin(selected_families)]

    # Filter by date range for display
    plot_df = df[(df["release_date"] >= pd.Timestamp(start_date)) &
                 (df["release_date"] <= pd.Timestamp(end_date))]

    horizon_col = f"{horizon_type}_estimate"
    ci_low_col = f"{horizon_type}_ci_low"
    ci_high_col = f"{horizon_type}_ci_high"

    fig = go.Figure()

    fitted_doubling_time = None
    current_horizon = None

    # Add scatter points with error bars for each family
    for family in plot_df["family"].unique():
        family_df = plot_df[plot_df["family"] == family]
        color = family_colors.get(family, "#999999")
        marker = family_markers.get(family, "circle")

        # Error bars (confidence intervals)
        error_y = None
        if show_ci:
            error_y = dict(
                type="data",
                symmetric=False,
                array=family_df[ci_high_col] - family_df[horizon_col],
                arrayminus=family_df[horizon_col] - family_df[ci_low_col],
                color=color,
                thickness=1.5,
                width=4,
            )

        hover_text = [
            f"<b>{row['model_name']}</b><br>"
            f"Release: {row['release_date'].strftime('%Y-%m-%d')}<br>"
            f"{horizon_type.upper()} Horizon: {format_hover_time(row[horizon_col])}<br>"
            f"95% CI: [{format_hover_time(row[ci_low_col])}, {format_hover_time(row[ci_high_col])}]<br>"
            f"Avg Score: {row['average_score']:.1%}<br>"
            f"SOTA: {'Yes' if row['is_sota'] else 'No'}"
            for _, row in family_df.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=family_df["release_date"],
            y=family_df[horizon_col],
            mode="markers",
            name=family,
            marker=dict(
                size=marker_size,
                color=color,
                symbol=marker,
                line=dict(width=1, color="white"),
            ),
            error_y=error_y,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text,
        ))

    # Add exponential trendline (fitted to data)
    if show_trendline and len(df) >= 3:
        try:
            reg, r_squared, fit_df = fit_exponential_trendline(df, horizon_col, after_date=fit_start_date)

            # Calculate doubling time
            fitted_doubling_time = np.log(2) / reg.coef_[0]  # in days
            doubling_months = fitted_doubling_time / 30.44

            # Get current horizon value (from latest model)
            max_date = fit_df["release_date"].max()
            current_horizon = fit_df.loc[fit_df["release_date"] == max_date, horizon_col].values[0]

            # Generate trendline points
            min_date = fit_df["release_date"].min()

            # Solid line for fitted range
            date_range = pd.date_range(min_date, max_date, periods=100)
            X_pred = np.array([d.toordinal() for d in date_range]).reshape(-1, 1)
            y_pred = np.exp(reg.predict(X_pred))

            fig.add_trace(go.Scatter(
                x=date_range,
                y=y_pred,
                mode="lines",
                name=f"Fitted Trend (R²={r_squared:.2f})",
                line=dict(color="#2563eb", width=2.5),
                customdata=[[d.strftime("%Y-%m-%d"), f"{v:.2f}"] for d, v in zip(date_range, y_pred)],
                hovertemplate=f"Date: %{{customdata[0]}}<br>Horizon: %{{customdata[1]}} hrs<br>Doubling time: {fitted_doubling_time:.0f} days ({doubling_months:.1f} months)<br>R²: {r_squared:.2f}<extra></extra>",
            ))

            # Dashed projection line (using fitted doubling time)
            if show_projection:
                proj_end = pd.Timestamp(end_date)
                if proj_end > max_date:
                    proj_range = pd.date_range(max_date, proj_end, periods=50)
                    X_proj = np.array([d.toordinal() for d in proj_range]).reshape(-1, 1)
                    y_proj = np.exp(reg.predict(X_proj))

                    # Add confidence band for projection
                    days_from_data = np.array([(d - max_date).days for d in proj_range])
                    uncertainty_factor = 1 + 0.002 * days_from_data

                    fig.add_trace(go.Scatter(
                        x=proj_range,
                        y=y_proj * uncertainty_factor**2,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                    fig.add_trace(go.Scatter(
                        x=proj_range,
                        y=y_proj / uncertainty_factor**2,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(37, 99, 235, 0.15)",
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                    fig.add_trace(go.Scatter(
                        x=proj_range,
                        y=y_proj,
                        mode="lines",
                        name=f"Projection ({fitted_doubling_time:.0f}d doubling)",
                        line=dict(color="#2563eb", width=2, dash="dash"),
                        opacity=0.6,
                        customdata=[[d.strftime("%Y-%m-%d"), f"{v:.2f}"] for d, v in zip(proj_range, y_proj)],
                        hovertemplate="Date: %{customdata[0]}<br>Projected horizon: %{customdata[1]} hrs<extra></extra>",
                    ))

            # Add annotation for fitted doubling time
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"<b>Fitted Trend</b><br>Doubling: {fitted_doubling_time:.0f} days ({doubling_months:.1f} mo)<br>R²: {r_squared:.2f}",
                showarrow=False,
                font=dict(size=11, color="#2563eb"),
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#2563eb",
                borderwidth=1,
                borderpad=6,
            )

            # Add custom doubling time projections
            if custom_doubling_times and show_projection:
                proj_end = pd.Timestamp(end_date)
                start_value = y_pred[-1]  # Start from end of fitted line

                for i, custom_dt in enumerate(custom_doubling_times):
                    if custom_dt and custom_dt > 0:
                        color = custom_colors[i % len(custom_colors)]
                        custom_months = custom_dt / 30.44

                        proj_range, y_custom = generate_custom_projection(
                            max_date, start_value, custom_dt, proj_end, 50
                        )

                        fig.add_trace(go.Scatter(
                            x=proj_range,
                            y=y_custom,
                            mode="lines",
                            name=f"Custom: {custom_dt:.0f}d ({custom_months:.1f}mo)",
                            line=dict(color=color, width=2, dash="dot"),
                            customdata=[[d.strftime("%Y-%m-%d"), f"{v:.2f}", custom_dt, custom_months]
                                       for d, v in zip(proj_range, y_custom)],
                            hovertemplate="Date: %{customdata[0]}<br>Horizon: %{customdata[1]} hrs<br>Doubling: %{customdata[2]:.0f} days (%{customdata[3]:.1f} months)<extra></extra>",
                        ))

        except Exception as e:
            st.warning(f"Could not fit trendline: {e}")

    # Configure axes
    success_pct = "50" if horizon_type == "p50" else "80"

    fig.update_layout(
        title=dict(
            text=f"Frontier AI's Software R&D Capabilities<br><sup>Task duration (for humans) that model completes with {success_pct}% success rate</sup>",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Model Release Date",
            tickformat="%Y",
            dtick="M12",
            ticklabelmode="period",
            gridcolor="rgba(128,128,128,0.2)",
            range=[start_date, end_date],
        ),
        yaxis=dict(
            title="Time Horizon (hours)",
            type="log",
            gridcolor="rgba(128,128,128,0.2)",
            # Values in hours
            tickvals=[1/3600, 1/1800, 1/720, 1/360, 1/120, 1/60, 1/30, 1/12, 1/6, 0.5, 1, 2, 4, 8, 16, 24, 48, 168, 720],
            ticktext=["1s", "2s", "5s", "10s", "30s", "1m", "2m", "5m", "10m", "30m", "1h", "2h", "4h", "8h", "16h", "1d", "2d", "1w", "1mo"],
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        hovermode="closest",
        plot_bgcolor="white",
        height=650,
        margin=dict(r=180),
    )

    # Add vertical line for "today"
    today = pd.Timestamp.now()
    if pd.Timestamp(start_date) <= today <= pd.Timestamp(end_date):
        fig.add_shape(
            type="line",
            x0=today, x1=today,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(255,0,255,0.5)", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=today,
            y=1.02,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(size=10, color="rgba(255,0,255,0.7)"),
        )

    return fig, fitted_doubling_time, current_horizon


def create_score_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart comparing model average scores."""
    df_sorted = df.sort_values("average_score", ascending=True).tail(15)

    fig = go.Figure()

    family_colors = {
        "Anthropic": "#d97757",
        "OpenAI": "#18a683",
        "DeepSeek": "#5B8FF9",
        "Google": "#F6BD16",
        "Alibaba": "#E8684A",
        "xAI": "#9C5EDA",
        "Moonshot": "#FF9D4D",
        "Other": "#999999",
    }

    colors = [family_colors.get(f, "#999999") for f in df_sorted["family"]]

    fig.add_trace(go.Bar(
        y=df_sorted["model_name"],
        x=df_sorted["average_score"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1%}" for s in df_sorted["average_score"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Average Score: %{x:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title="Top 15 Models by Average Score",
        xaxis=dict(
            title="Average Score",
            tickformat=".0%",
            range=[0, 1],
        ),
        yaxis=dict(title=""),
        height=500,
        margin=dict(l=200),
        showlegend=False,
    )

    return fig


def main():
    st.title("📊 METR Benchmark Results Dashboard")
    st.markdown("""
    This dashboard displays AI model capability horizons from the
    [METR-Horizon-v1 benchmark](https://metr.org), showing how long tasks AI agents
    can successfully complete compared to human experts.
    """)

    # Fetch data
    with st.spinner("Fetching benchmark data..."):
        try:
            raw_data = fetch_benchmark_data()
            df = parse_results_to_dataframe(raw_data)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    # Sidebar controls
    st.sidebar.header("📈 Plot Settings")

    horizon_type = st.sidebar.radio(
        "Horizon Type",
        options=["p50", "p80"],
        format_func=lambda x: "50% Success (Median)" if x == "p50" else "80% Success",
        index=0,
    )

    all_families = sorted(df["family"].unique())
    selected_families = st.sidebar.multiselect(
        "Model Families",
        options=all_families,
        default=all_families,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Trendline & Projections")

    show_trendline = st.sidebar.checkbox("Show Fitted Trendline", value=True)
    show_ci = st.sidebar.checkbox("Show Confidence Intervals", value=True)
    show_projection = st.sidebar.checkbox("Show Projections", value=True)

    fit_start_date = st.sidebar.date_input(
        "Fit data from",
        value=pd.Timestamp("2020-01-01"),
        min_value=pd.Timestamp("2019-01-01"),
        max_value=pd.Timestamp("2025-01-01"),
        help="Only include models released after this date in the trendline fit"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Custom Doubling Times")
    st.sidebar.caption("Add your own doubling time scenarios to compare")

    custom_doubling_times = []

    col1, col2 = st.sidebar.columns(2)
    custom_dt_1 = col1.number_input(
        "Scenario 1 (days)",
        min_value=0,
        max_value=1000,
        value=0,
        help="Enter 0 to disable"
    )
    custom_dt_2 = col2.number_input(
        "Scenario 2 (days)",
        min_value=0,
        max_value=1000,
        value=0,
        help="Enter 0 to disable"
    )

    col3, col4 = st.sidebar.columns(2)
    custom_dt_3 = col3.number_input(
        "Scenario 3 (days)",
        min_value=0,
        max_value=1000,
        value=0,
        help="Enter 0 to disable"
    )

    # Quick presets
    st.sidebar.caption("Quick presets:")
    preset_col1, preset_col2, preset_col3 = st.sidebar.columns(3)

    if custom_dt_1 > 0:
        custom_doubling_times.append(custom_dt_1)
    if custom_dt_2 > 0:
        custom_doubling_times.append(custom_dt_2)
    if custom_dt_3 > 0:
        custom_doubling_times.append(custom_dt_3)

    col1, col2 = st.sidebar.columns(2)
    start_year = col1.number_input("Start Year", min_value=2018, max_value=2027, value=2019)
    end_year = col2.number_input("End Year", min_value=2020, max_value=2035, value=2028)

    # Main plot
    st.subheader("Time Horizon Over Time")

    fig, fitted_doubling_time, current_horizon = create_horizon_plot(
        df,
        horizon_type=horizon_type,
        show_trendline=show_trendline,
        show_ci=show_ci,
        selected_families=selected_families,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-01-01",
        show_projection=show_projection,
        fit_start_date=str(fit_start_date),
        custom_doubling_times=custom_doubling_times if custom_doubling_times else None,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Milestone Calculator
    if fitted_doubling_time and current_horizon:
        st.subheader("🎯 Milestone Calculator")
        st.markdown("Calculate when AI will reach specific capability milestones based on different doubling time assumptions.")

        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            target_hours = st.selectbox(
                "Target Horizon",
                options=[1, 4, 8, 24, 48, 168, 720],
                format_func=lambda x: {
                    1: "1 hour",
                    4: "4 hours",
                    8: "8 hours (1 workday)",
                    24: "1 day",
                    48: "2 days",
                    168: "1 week",
                    720: "1 month",
                }[x],
                index=2,
            )
            target_value_hours = target_hours  # Already in hours

        with calc_col2:
            st.markdown("**Current state:**")
            st.markdown(f"- Latest horizon: **{format_hover_time(current_horizon)}**")
            st.markdown(f"- Fitted doubling time: **{fitted_doubling_time:.0f} days** ({fitted_doubling_time/30.44:.1f} months)")

        # Calculate for fitted and custom doubling times
        latest_date = df["release_date"].max()

        all_scenarios = [("Fitted trend", fitted_doubling_time, "#2563eb")]
        custom_colors = ["#e11d48", "#7c3aed", "#059669"]
        for i, cdt in enumerate(custom_doubling_times):
            if cdt > 0:
                all_scenarios.append((f"Custom ({cdt}d)", cdt, custom_colors[i % len(custom_colors)]))

        st.markdown(f"**When will AI reach {target_hours} hour{'s' if target_hours > 1 else ''} horizon?**")

        result_cols = st.columns(len(all_scenarios))
        for i, (name, dt, color) in enumerate(all_scenarios):
            with result_cols[i]:
                if current_horizon < target_value_hours:
                    target_date = calculate_time_to_horizon(current_horizon, target_value_hours, dt, latest_date)
                    days_away = (target_date - pd.Timestamp.now()).days
                    st.markdown(f"**{name}:**")
                    st.markdown(f"📅 {target_date.strftime('%B %Y')}")
                    if days_away > 0:
                        st.caption(f"({days_away} days from now)")
                    else:
                        st.caption("(Already passed)")
                else:
                    st.markdown(f"**{name}:**")
                    st.markdown("✅ Already achieved!")

    # Doubling Time Calculator
    st.subheader("📐 Doubling Time Calculator")
    st.markdown("Calculate the implied doubling time between two models or two dates.")

    calc_mode = st.radio(
        "Calculate between:",
        options=["Two Models", "Two Dates"],
        horizontal=True,
    )

    horizon_col = f"{horizon_type}_estimate"

    if calc_mode == "Two Models":
        model_col1, model_col2 = st.columns(2)

        # Sort models by release date for the dropdown (newest first)
        models_sorted = df.sort_values("release_date", ascending=False)
        model_names = models_sorted["model_name"].tolist()

        with model_col1:
            model1_name = st.selectbox(
                "Earlier Model",
                options=model_names,
                index=len(model_names) - 1,  # Default to oldest
                key="model1"
            )
            model1_data = df[df["model_name"] == model1_name].iloc[0]
            st.caption(f"Released: {model1_data['release_date'].strftime('%Y-%m-%d')}")
            st.caption(f"{horizon_type.upper()} Horizon: {format_hover_time(model1_data[horizon_col])}")

        with model_col2:
            model2_name = st.selectbox(
                "Later Model",
                options=model_names,
                index=0,  # Default to newest
                key="model2"
            )
            model2_data = df[df["model_name"] == model2_name].iloc[0]
            st.caption(f"Released: {model2_data['release_date'].strftime('%Y-%m-%d')}")
            st.caption(f"{horizon_type.upper()} Horizon: {format_hover_time(model2_data[horizon_col])}")

        # Calculate doubling time between the two models
        date1 = model1_data["release_date"]
        date2 = model2_data["release_date"]
        horizon1 = model1_data[horizon_col]
        horizon2 = model2_data[horizon_col]

        if date1 != date2 and horizon1 > 0 and horizon2 > 0 and horizon2 != horizon1:
            days_between = (date2 - date1).days
            # doubling_time = days_between / log2(horizon2/horizon1)
            if horizon2 > horizon1:
                log_ratio = np.log2(horizon2 / horizon1)
                implied_doubling = days_between / log_ratio
                doubling_months = implied_doubling / 30.44

                st.success(f"""
                **Implied Doubling Time: {implied_doubling:.0f} days ({doubling_months:.1f} months)**

                - Time between releases: {days_between} days
                - Horizon improvement: {horizon1:.2f} hrs → {horizon2:.2f} hrs ({horizon2/horizon1:.1f}x)
                """)
            else:
                st.warning("The later model has a lower horizon than the earlier model. Select models where capability increased.")
        elif date1 == date2:
            st.info("Please select two different models with different release dates.")
        else:
            st.info("Cannot calculate doubling time with the selected models.")

    else:  # Two Dates mode
        if fitted_doubling_time:
            date_col1, date_col2 = st.columns(2)

            min_date = df["release_date"].min().date()
            max_date = df["release_date"].max().date()

            with date_col1:
                start_calc_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="start_calc_date"
                )

            with date_col2:
                end_calc_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="end_calc_date"
                )

            if start_calc_date < end_calc_date:
                # Find models closest to these dates
                df_sorted = df.sort_values("release_date")

                # Find closest model to start date
                start_ts = pd.Timestamp(start_calc_date)
                end_ts = pd.Timestamp(end_calc_date)

                models_before_end = df_sorted[df_sorted["release_date"] <= end_ts]
                models_after_start = df_sorted[df_sorted["release_date"] >= start_ts]

                if len(models_before_end) > 0 and len(models_after_start) > 0:
                    # Get the first model after start and last model before end
                    first_model = models_after_start.iloc[0]
                    last_model = models_before_end.iloc[-1]

                    if first_model["release_date"] < last_model["release_date"]:
                        days_between = (last_model["release_date"] - first_model["release_date"]).days
                        horizon1 = first_model[horizon_col]
                        horizon2 = last_model[horizon_col]

                        if horizon2 > horizon1:
                            log_ratio = np.log2(horizon2 / horizon1)
                            implied_doubling = days_between / log_ratio
                            doubling_months = implied_doubling / 30.44

                            st.success(f"""
                            **Implied Doubling Time: {implied_doubling:.0f} days ({doubling_months:.1f} months)**

                            - Using models: **{first_model['model_name']}** → **{last_model['model_name']}**
                            - Dates: {first_model['release_date'].strftime('%Y-%m-%d')} → {last_model['release_date'].strftime('%Y-%m-%d')} ({days_between} days)
                            - Horizon improvement: {horizon1:.2f} hrs → {horizon2:.2f} hrs ({horizon2/horizon1:.1f}x)
                            """)
                        else:
                            st.warning("No capability improvement detected in this date range.")
                    else:
                        st.info("Not enough models in the selected date range.")
                else:
                    st.info("No models found in the selected date range.")
            else:
                st.info("End date must be after start date.")
        else:
            st.info("Enable the trendline to use date-based calculations.")

    # Additional charts and data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Rankings")
        score_fig = create_score_comparison_chart(df)
        st.plotly_chart(score_fig, use_container_width=True)

    with col2:
        st.subheader("Latest SOTA Models")
        sota_df = df[df["is_sota"]].sort_values("release_date", ascending=False).head(10)
        display_cols = ["model_name", "release_date", "p50_estimate", "average_score", "family"]
        sota_display = sota_df[display_cols].copy()
        sota_display["release_date"] = sota_display["release_date"].dt.strftime("%Y-%m-%d")
        sota_display["p50_estimate"] = sota_display["p50_estimate"].apply(format_time_label)
        sota_display["average_score"] = sota_display["average_score"].apply(lambda x: f"{x:.1%}")
        sota_display.columns = ["Model", "Release Date", "P50 Horizon", "Avg Score", "Family"]
        st.dataframe(sota_display, use_container_width=True, hide_index=True)

    # Data table
    with st.expander("📋 View All Data"):
        display_df = df.copy()
        display_df["release_date"] = display_df["release_date"].dt.strftime("%Y-%m-%d")
        for col in ["p50_estimate", "p50_ci_low", "p50_ci_high", "p80_estimate", "p80_ci_low", "p80_ci_high"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["average_score"] = display_df["average_score"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        display_df["cost_usd"] = display_df["cost_usd"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")

        st.dataframe(
            display_df[["model_name", "family", "release_date", "p50_estimate", "p80_estimate",
                       "average_score", "is_sota", "cost_usd"]],
            use_container_width=True,
            hide_index=True,
        )

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name="metr_benchmark_results.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source:** [METR Benchmark Results](https://metr.org/assets/benchmark_results.yaml)
    **Benchmark:** METR-Horizon-v1
    **Last Updated:** Data is fetched dynamically and cached for 1 hour
    """)


if __name__ == "__main__":
    main()
