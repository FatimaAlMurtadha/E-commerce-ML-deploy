"""Reusable chart styling and prediction gauge components."""

import plotly.graph_objects as go
import streamlit as st

from frontend.ui_config import (
    BLUE,
    FONT_FAMILY,
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    SEQ_BLUE,
)


def style_fig(fig: go.Figure, height: int = 320, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PRIMARY_INK, family=FONT_FAMILY, size=13),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font=dict(color=PRIMARY_INK, family=FONT_FAMILY)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRIDLINE, color=MUTED_INK)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, color=MUTED_INK)
    return fig


def make_gauge(probability_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number={"suffix": "%", "font": {"size": 42, "color": PRIMARY_INK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED_INK, "tickfont": {"color": MUTED_INK}},
                "bar": {"color": BLUE, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": SEQ_BLUE[0]},
                    {"range": [33, 66], "color": SEQ_BLUE[2]},
                    {"range": [66, 100], "color": SEQ_BLUE[4]},
                ],
                "threshold": {"line": {"color": PRIMARY_INK, "width": 2}, "thickness": 0.75, "value": 50},
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PRIMARY_INK, "family": FONT_FAMILY},
    )
    return fig
