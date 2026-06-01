# Fund Return AutoML App
# Streamlit app for classifying next-period up/down movement from return or price data.
# Run "python -m streamlit run app.py" in terminal

# ----- SETUP -----
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(
    page_title="Fund Return AutoML",
    page_icon="📈",
    layout="wide"
)

SAMPLE_PATH = Path("data") / "S&P 500 Historical Data.csv" # sample data included in repo


# ----- Custom functions -----

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert numeric strings with commas, percent signs, and dollar signs into numeric values."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_return_data(raw_df: pd.DataFrame, date_col: str, value_col: str, input_mode: str) -> pd.DataFrame:
    """Create a clean dataframe with date and decimal return columns."""
    df = raw_df[[date_col, value_col]].copy()
    df.columns = ["date", "input_value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce") # fill non-date column with NaT
    df["input_value"] = clean_numeric_column(df["input_value"]) # perform numeric string cleaning
    df = df.dropna(subset=["date", "input_value"]).sort_values("date") #remove missing rows

    if input_mode == "Return":
        # If the source column contains percent signs or looks like percent returns, convert it into decimal returns.
        raw_as_text = raw_df[value_col].astype(str)
        has_percent_sign = raw_as_text.str.contains("%", regex=False).any()
        looks_like_percent = df["input_value"].abs().median() > 1
        df["return"] = df["input_value"] / 100 if has_percent_sign or looks_like_percent else df["input_value"]
    else:
        # If the user gives a price column, compute periodic returns from price.
        df["return"] = df["input_value"].pct_change()

    df = df[["date", "return"]].dropna()
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create return-based features and the next-period up/down target."""
    data = df.copy()

    # Lag features: recent return history.
    for lag in [1, 2, 3, 5]:
        data[f"return_lag_{lag}"] = data["return"].shift(lag)

    # Rolling average return: short-term trend.
    for window in [3, 5, 10]:
        data[f"rolling_mean_{window}"] = data["return"].rolling(window).mean()

    # Rolling volatility: short-term risk / variability.
    for window in [3, 5, 10]:
        data[f"rolling_vol_{window}"] = data["return"].rolling(window).std()

    # Momentum: cumulative return over recent periods.
    for window in [3, 5, 10]:
        data[f"momentum_{window}"] = data["return"].rolling(window).sum()

    # Simple direction flags.
    data["positive_return_lag_1"] = (data["return"].shift(1) > 0).astype(int)
    data["negative_return_lag_1"] = (data["return"].shift(1) < 0).astype(int)

    # Target: whether the next period return is positive.
    data["next_period_return"] = data["return"].shift(-1)
    data["target"] = (data["next_period_return"] > 0).astype(int)

    return data


def get_models() -> dict:
    """Define the four classification models used in the AutoML benchmark."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=61),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=61),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=61),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=61),
    }


def run_models(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict]:
    """Train each model and collect metrics, predictions, and fitted pipelines."""
    results = []
    fitted = {}
    models = get_models()

    status = st.status("Running AutoML benchmark...", expanded=True)

    for name, model in models.items():
        status.write(f"Training {name}...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc,
        })

        fitted[name] = {
            "pipeline": pipeline,
            "prediction": y_pred,
            "probability_up": y_prob,
        }

    status.update(label="AutoML complete", state="complete", expanded=False)
    leaderboard = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    return leaderboard, fitted


def feature_group_dict() -> dict:
    """Feature groups shown in the Feature Engineering section."""
    return {
        "Lagged Returns": [["return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5"], "Recent return history for 1, 2, 3, and 5 periods."],
        "Rolling Return": [["rolling_mean_3", "rolling_mean_5", "rolling_mean_10"], "Short-term average return over recent periods for 3, 5, and 10 periods."],
        "Rolling Volatility": [["rolling_vol_3", "rolling_vol_5", "rolling_vol_10"], "Recent variability in returns for 3, 5, and 10 periods."],
        "Momentum": [["momentum_3", "momentum_5", "momentum_10"], "Cumulative return over recent periods for 3, 5, and 10 periods."],
        "Direction Flags": [["positive_return_lag_1", "negative_return_lag_1"], "Whether the previous period was positive or negative"],
    }

# Performance Metric Interpretations
def interpret_accuracy(v):
    if v <= 0.50: return f"Below 0.50: no better than a coin flip; the model isn't learning the problem."
    if v <= 0.65: return f"Between 0.50 and 0.65: barely above chance; only {v*100:.1f}% of predictions are correct."
    if v <= 0.75: return f"Between 0.65 and 0.75: passable ({v*100:.1f}% correct), but confirm this isn't driven by class imbalance."
    if v <= 0.85: return f"Between 0.75 and 0.85: solid overall; {v*100:.1f}% of predictions are correct."
    if v <= 0.92: return f"Between 0.85 and 0.92: strong performance; {v*100:.1f}% correct across both classes."
    return         f"Above 0.92: outstanding ({v*100:.1f}% correct); verify no data leakage in your test set."
def interpret_f1(v):
    if v <= 0.60: return f"Below 0.60: poor balance of precision and recall; the model misses too many positives or raises too many false alarms."
    if v <= 0.75: return f"Between 0.60 and 0.75: fair precision/recall trade-off; acceptable for low-stakes use cases."
    if v <= 0.85: return f"Between 0.75 and 0.85: good balance; misses and false alarms are both manageable."
    return         f"Above 0.85: strong precision and recall; validate thoroughly as this level is uncommon."
def interpret_auc(v):
    if v <= 0.50: return f"Below 0.50: worse than random; check for a label-encoding bug or data leakage."
    if v <= 0.70: return f"Between 0.50 and 0.70: poor discrimination; the model barely separates positives from negatives."
    if v <= 0.80: return f"Between 0.70 and 0.80: fair separation between classes; useful as a baseline."
    if v <= 0.90: return f"Between 0.80 and 0.90: good class separation; you have flexibility in choosing a decision threshold."
    return         f"Above 0.90: outstanding class separation; reconfirm no train/test leakage before shipping."


def make_model_template(selected_model: str, selected_features: list[str], input_mode: str) -> str:
    """Create a simple Python template for reproducing the selected model."""
    model_code = {
        "Logistic Regression": "LogisticRegression(max_iter=1500, random_state=61)",
        "Random Forest Classifier": "RandomForestClassifier(n_estimators=300, random_state=61)",
        "Gradient Boosting Classifier": "GradientBoostingClassifier(random_state=61)",
        "MLP Classifier": "MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=61)",
    }[selected_model]

    feature_list = "[\n" + "\n".join([f'    "{feature}",' for feature in selected_features]) + "\n]"

    return f'''# Reproducible model template from the Streamlit AutoML app
# Selected model: {selected_model}
# Input mode used in app: {input_mode}

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def clean_numeric_column(series):
    # Converts finance-style strings such as "1,234.5" or "0.75%" into numbers.
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def create_features(df):
    # This assumes df has columns: date, return.
    data = df.copy().sort_values("date")

    for lag in [1, 2, 3, 5]:
        data[f"return_lag_{{lag}}"] = data["return"].shift(lag)

    for window in [3, 5, 10]:
        data[f"rolling_mean_{{window}}"] = data["return"].rolling(window).mean()
        data[f"rolling_vol_{{window}}"] = data["return"].rolling(window).std()
        data[f"momentum_{{window}}"] = data["return"].rolling(window).sum()

    data["positive_return_lag_1"] = (data["return"].shift(1) > 0).astype(int)
    data["negative_return_lag_1"] = (data["return"].shift(1) < 0).astype(int)

    # Target is next-period direction.
    data["next_period_return"] = data["return"].shift(-1)
    data["target"] = (data["next_period_return"] > 0).astype(int)

    return data


# Replace this filename and column names with your own dataset details.
raw = pd.read_csv("S&P 500 Historical Data.csv")
raw["date"] = pd.to_datetime(raw["Date"], errors="coerce")

# Option A: If using a return column such as "Change %".
raw["return"] = clean_numeric_column(raw["Change %"]) / 100

# Option B: If using a price column such as "Open", comment out Option A and use this instead.
# raw["price"] = clean_numeric_column(raw["Open"])
# raw = raw.sort_values("date")
# raw["return"] = raw["price"].pct_change()

model_df = create_features(raw[["date", "return"]]).dropna()

selected_features = {feature_list}

X = model_df[selected_features]
y = model_df["target"]

# Chronological split: earlier observations train, later observations test.
split_index = int(len(model_df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", {model_code})
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
prob_up = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_up))
'''


# -------------------------------------
# Streamlit App 
# -------------------------------------

# ----- Title Page -----
st.title("Fund Return AutoML")
st.caption("Upload periodic return or price data, create return-based features, compare classification models, and extract a reproducible model template.")

st.markdown(
    """
    This app creates AutoML models for predicting whether the **next period return** is up or down.  
    It is designed for simple model comparison and visual interpretation rather than live trading.
    """
)


# ----- 1. Data Upload -----

st.header("1. Data Upload")
st.write("Upload a CSV file with a date column and a return or a price column.")
st.caption("The app allows for choosing appropriate columns and handling common formats.")

# Use two columns so the upload action and sample-data checkbox sit near each other
upload_col, sample_col = st.columns([2.5, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

with sample_col:
    use_sample = st.checkbox("Use S&P 500 2025 returns")

# If the checkbox is selected, the app uses the repo sample file. Otherwise, it defaults for the user to upload a CSV.
if use_sample:
    raw_df = pd.read_csv(SAMPLE_PATH)
    st.success(f"Loaded sample file (source: https://www.investing.com/indices/us-spx-500-historical-data)")
else:
    if uploaded_file is None:
        st.info("Upload a CSV file to begin, or select the sample S&P 500 dataset.")
        st.stop()

    raw_df = pd.read_csv(uploaded_file)

# Data Preview
st.subheader("Data Preview")
st.dataframe(raw_df.head(10), use_container_width=True)

# Column Selection
columns = raw_df.columns.tolist()

## Date
default_date_index = 0
for i, col in enumerate(columns):
    if "date" in col.lower():
        default_date_index = i
        break
date_col = st.selectbox("Select date column", columns, index=default_date_index)

## Value
value_col = st.selectbox("Select value column", columns, index=len(columns) - 1)
input_mode = st.radio(
    "Value Type",
    ["Return", "Price"],
    help="Use Return for values like 0.012 or 1.2%. Use Price for dollar values like Open, Close, or adjusted price.",
    horizontal=True,
)
if input_mode == "Price":
    st.caption(
        "Price column will be converted to returns using percent change calculation. "
        "The percent change for the first date will be NaN and will be dropped in the validation step. "
    )

# Clean data
return_df = prepare_return_data(raw_df, date_col, value_col, input_mode)

# Data Validation
st.subheader("Validated data format")
st.dataframe(return_df.head(10), use_container_width=True)

# Key Stats as cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean Rows", f"{len(return_df):,}")
c2.metric("Average Return", f"{return_df['return'].mean():.4%}")
c3.metric("Volatility", f"{return_df['return'].std():.4%}")
c4.metric("Positive Return Periods", f"{(return_df['return'] > 0).mean():.1%}")

# Missing records
excluded_rows = ~raw_df.index.isin(return_df.index)
excluded_df = raw_df.loc[excluded_rows]
## if "Price" is selected, the first record gets dropped but does not show up in excluded_df to avoid alarming the user.
if input_mode == "Price" and len(excluded_df) > 0:
    first_date_index = pd.to_datetime(raw_df[date_col], errors="coerce").sort_values().index[0]
    excluded_df = excluded_df.drop(index=first_date_index, errors="ignore")

# Summary Stats & Missing records present
summary_stats = return_df["return"].describe().to_frame("return")
with st.expander("Returns Summary Stats & Excluded Records"):
    st.write("Summary Statistics on Returns after Cleaning")
    st.dataframe(summary_stats, use_container_width=True)
    st.markdown("**Excluded Records**")
    if excluded_df.empty:
        st.success("No records were excluded after cleaning.")
    else:
        st.dataframe(excluded_df, use_container_width=True)

# reset index after cleaning & missing record extract is done
return_df = return_df.reset_index(drop=True)


# ----- 2. Feature Engineering -----

st.header("2. Feature Engineering")
st.write("The app creates standard return-based features from past observations. These features are used in AutoML to predict next-period direction.")

feature_df = create_features(return_df)
feature_groups = feature_group_dict()
all_features = [feature for group in feature_groups.values() for feature in group[0]]   # takes feature names
all_definitions = [group[1] for group in feature_groups.values()]   # takes feature definitions

selected_features = []

with st.expander("Feature Explanations & Custom Selection", expanded=False):

    use_all_features = st.checkbox("Select all features", value=True)
    if use_all_features:
        selected_features = all_features
        for group_name, features in feature_groups.items():
            st.markdown(f"**{group_name}**")
            st.caption(features[1])  # Display the definition
    else:
        for group_name, features in feature_groups.items():
            st.markdown(f"**{group_name}**")
            st.caption(features[1])  # Display the definition
            chosen = st.multiselect(
                "select",
                features[0],
                default=features[0],
                key=f"features_{group_name}",
                label_visibility="collapsed",
            )
            selected_features.extend(chosen)

if not selected_features:
    st.warning("Select at least one feature to continue.")
    st.stop()

model_df = feature_df[["date", "return", "next_period_return", "target"] + selected_features].dropna().reset_index(drop=True)

st.write(f"Number of Features: **{len(selected_features)}**")
st.dataframe(model_df[["date", "return", "target"] + selected_features].head(10), use_container_width=True)


# ----- 3. ML Modeling -----

st.header("3. ML Modeling")
st.write("Perform a chronological split: earlier observations train the models, and later observations test the models.")

with st.expander("Chronological Split", expanded=False):
    st.write("For time-series return data, the model should train on earlier observations and test on later observations. This avoids mixing future observations into the training period.")
    train_pct = st.slider("Training data percentage (%)", min_value=60, max_value=95, value=80, step=1)

# Split Visualization
split_index = int(len(model_df) * train_pct / 100)
train_count = split_index
test_count = len(model_df) - split_index
## Actual Date points
date_series = pd.to_datetime(model_df["date"]).reset_index(drop=True)
train_start = date_series.iloc[0]
train_end = date_series.iloc[split_index - 1]
test_start = date_series.iloc[split_index]
test_end = date_series.iloc[-1]
train_mid = train_start + (train_end - train_start) / 2
test_mid = test_start + (test_end - test_start) / 2
## Plot Configuration
split_fig = go.Figure()
split_fig.add_trace(go.Scatter(x=[train_start, test_end], y=[0, 0], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False)) # invisible trace to set x-axis range
split_fig.add_shape(type="rect", x0=train_start, x1=train_end, y0=0.25, y1=0.75, line=dict(width=0), fillcolor="rgba(31, 119, 180, 0.65)") # train bar
split_fig.add_shape(type="rect", x0=test_start, x1=test_end, y0=0.25, y1=0.75, line=dict(width=0), fillcolor="rgba(31, 119, 180, 0.35)") # test bar
split_fig.add_shape(type="line", x0=train_end, x1=train_end, y0=0.15, y1=0.85, line=dict(dash="dash", width=2)) # split bar
## Plot labels
split_fig.add_annotation(x=train_mid, y=0.5, text=f"Train: {train_count} records", showarrow=False, font=dict(size=13)) # train label
split_fig.add_annotation(x=test_mid, y=0.5, text=f"Test: {test_count} records", showarrow=False, font=dict(size=13)) # test label
split_fig.add_annotation(x=train_end, y=0.95, text=f"Last training date: {train_end.strftime('%Y-%m-%d')}", showarrow=False, yanchor="bottom", font=dict(size=12)) # split date label
split_fig.update_layout(height=190, margin=dict(l=10, r=10, t=45, b=20), showlegend=False, 
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]), 
                        xaxis=dict(type="date", title=None, showgrid=False)) # overall declutter layout
## show plot
st.plotly_chart(split_fig, use_container_width=True)


# Actual Splitting
X = model_df[selected_features]
y = model_df["target"]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
## 0 split handling
if len(X_train) == 0 or len(X_test) == 0:
    st.warning("The split does not leave enough data for both training and testing.")
    st.stop()

# ML Modeling
st.space("large")
st.write("Build four ML models and compare their performance.")
run_button = st.button("Run AutoML", type="primary")

if run_button:
    leaderboard, fitted_models = run_models(X_train, X_test, y_train, y_test)
    st.session_state["leaderboard"] = leaderboard
    st.session_state["fitted_models"] = fitted_models
    st.session_state["model_df"] = model_df
    st.session_state["selected_features"] = selected_features
    st.session_state["split_index"] = split_index
    st.session_state["y_test"] = y_test
    st.session_state["input_mode"] = input_mode
elif "leaderboard" not in st.session_state:
    st.info("Click 'Run AutoML' to train and compare the models.")
    st.stop()
else:
    # Keep previous results visible when the user changes diagnostic dropdowns.
    leaderboard = st.session_state["leaderboard"]
    fitted_models = st.session_state["fitted_models"]
    model_df = st.session_state["model_df"]
    selected_features = st.session_state["selected_features"]
    split_index = st.session_state["split_index"]
    y_test = st.session_state["y_test"]
    input_mode = st.session_state["input_mode"]


# ----- 4. Assessment -----

st.header("4. Assessment")

# Leaderboard chart
st.subheader("Comparison leaderboard")
st.write("The leaderboard compares four classification models using Accuracy, F1, and ROC-AUC.")
st.dataframe(
    leaderboard.style.format({"Accuracy": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}),
    use_container_width=True,
)

# Validation Metric bar chart
metric_choice = st.selectbox("Model Comparison by", ["F1", "Accuracy", "ROC-AUC"], index=0)
bar_fig = px.bar(
    leaderboard.sort_values(metric_choice, ascending=True),
    x=metric_choice,
    y="Model",
    orientation="h",
    labels={metric_choice: "", "Model": ""}
)
bar_fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=20))
st.plotly_chart(bar_fig, use_container_width=True)

# Diagnostics Visuals
st.subheader("Model Diagnostics")

model_order = leaderboard["Model"].tolist()
name_cols = st.columns(4)
for col, model_name in zip(name_cols, model_order):
    with col:
        st.markdown(f"**{model_name}**")

## Confusion Matrix
st.space("small")
st.write("**Confusion Matrix**")
cm_cols = st.columns(4)

for col, model_name in zip(cm_cols, model_order):
    with col:
        selected_output = fitted_models[model_name]
        cm = confusion_matrix(y_test, selected_output["prediction"], labels=[0, 1])

        cm_fig = px.imshow(cm, x=["Pred. Down", "Pred. Up"], y=["Actual Down", "Actual Up"], text_auto=True)
        cm_fig.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(cm_fig, use_container_width=True)

        true_down, false_up = cm[0, 0], cm[0, 1]
        false_down, true_up = cm[1, 0], cm[1, 1]
st.caption("Confusion matrix shows the counts of correct and incorrect classifications.")
st.caption("The higher upper left (correct down) and lower right (correct up), the better predictive performance.")

## ROC Curve
st.space("small")
st.write("**ROC Curve**")
roc_cols = st.columns(4)

for col, model_name in zip(roc_cols, model_order):
    with col:
        selected_output = fitted_models[model_name]

        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, selected_output["probability_up"])
            auc_value = roc_auc_score(y_test, selected_output["probability_up"])

            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_value:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
            roc_fig.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis_title=None, yaxis_title=None, showlegend=False)
            st.plotly_chart(roc_fig, use_container_width=True)
            st.caption(f"AUC: {auc_value:.3f}")
        else:
            st.info("ROC not available.")
st.caption("ROC curve shows the tradeoff between true positive rate and false positive rate across classification thresholds.")
st.caption("The higher the curve (above the diagonal) and AUC, the better predictive performance.")

## Feature Importance
st.space("small")
st.write("**Feature Importance**")
importance_cols = st.columns(4)

for col, model_name in zip(importance_cols, model_order):
    with col:
        if model_name in ["Random Forest Classifier", "Gradient Boosting Classifier"]:
            importance_pipeline = fitted_models[model_name]["pipeline"]
            importance_values = importance_pipeline.named_steps["model"].feature_importances_

            importance_df = pd.DataFrame({"Feature": selected_features, "Importance": importance_values})
            importance_df = importance_df.sort_values("Importance", ascending=False).head(12)

            importance_fig = px.bar(importance_df.sort_values("Importance", ascending=True), x="Importance", y="Feature", orientation="h")
            importance_fig.update_layout(height=320, margin=dict(l=5, r=5, t=5, b=5), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.markdown("<div style='height:300px'></div>", unsafe_allow_html=True)
st.caption("Feature importance shows which features contributed most to the model's predictions.")
st.caption("Only available for tree-based models.")


# ----- 5. Model Selection & Extract -----

st.header("5. Model Selection & Extract")
st.write("Select the final model based on the assessment results and export a reproducible model template.")

## Select model
final_model = st.selectbox("Select final model", leaderboard["Model"].tolist(), index=0, label_visibility="collapsed")
final_row = leaderboard[leaderboard["Model"] == final_model].iloc[0]

## Metric extract & interpretation
card1, card2, card3 = st.columns(3)
card1.metric("Accuracy", f"{final_row['Accuracy']:.3f}")
card1.caption(interpret_accuracy(final_row['Accuracy']))
card2.metric("F1 Score", f"{final_row['F1']:.3f}")
card2.caption(interpret_f1(final_row['F1']))
card3.metric("ROC-AUC", f"{final_row['ROC-AUC']:.3f}")
card3.caption(interpret_auc(final_row['ROC-AUC']))

# Extract Python template for selected model
model_template = make_model_template(final_model, selected_features, input_mode)
st.download_button(
    label="Download reproducible model template (.py)",
    data=model_template,
    file_name=f"{final_model.lower().replace(' ', '_')}_template.py",
    mime="text/x-python",
)