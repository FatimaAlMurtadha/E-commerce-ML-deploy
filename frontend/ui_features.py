"""Feature preparation used by the analytics view."""

import pandas as pd
import streamlit as st

from frontend.ui_config import MODEL_FEATURE_COLUMNS, WEEKDAY_MAP


@st.cache_data(show_spinner=False)
def build_session_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre-order clicks and carts into one row per session."""
    target = (
        events.groupby("session")["type"]
        .apply(lambda values: "orders" in values.values)
        .astype(int)
        .rename("target")
    )

    rows = []
    for session_id, session_events in events.groupby("session"):
        session_events = session_events.sort_values("ts")
        order_mask = session_events["type"] == "orders"
        first_order = order_mask.values.argmax() if order_mask.any() else len(session_events)
        feature_events = session_events.iloc[:first_order]
        feature_events = feature_events[feature_events["type"].isin(["clicks", "carts"])]
        if feature_events.empty:
            continue
        rows.append(
            {
                "session": session_id,
                "num_clicks": int((feature_events["type"] == "clicks").sum()),
                "num_carts": int((feature_events["type"] == "carts").sum()),
                "num_events": len(feature_events),
                "num_unique_items": feature_events["aid"].nunique(),
                "session_duration_seconds": (
                    feature_events["ts"].max() - feature_events["ts"].min()
                ) / 1000.0,
                "hour": int(feature_events["hour"].iloc[-1]),
                "weekday": feature_events["weekday"].iloc[-1],
            }
        )

    features = pd.DataFrame(rows).set_index("session")
    features["weekday"] = features["weekday"].map(WEEKDAY_MAP).fillna(0)
    return features[MODEL_FEATURE_COLUMNS].join(target)
