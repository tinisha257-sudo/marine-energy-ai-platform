import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Marine Energy AI Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0,190,255,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0,255,200,0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #031b2b 0%,
            #063b55 50%,
            #087b83 100%
        );
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #021321,
        #043b55
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 19px;
    color: #b9eaff;
    margin-bottom: 35px;
}

.section-title {
    font-size: 29px;
    font-weight: 800;
    color: #79d9ff;
    margin-top: 25px;
    margin-bottom: 18px;
}

.card {
    background: rgba(3,27,43,0.88);
    border: 1px solid rgba(120,220,255,0.22);
    border-radius: 18px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.20);
}

.card-title {
    color: #b9dced;
    font-size: 16px;
    font-weight: 600;
}

.card-value {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 12px;
}

.card-desc {
    color: #89b8ca;
    font-size: 13px;
    margin-top: 6px;
}

.ai-box {
    background: linear-gradient(
        135deg,
        rgba(0,110,150,0.55),
        rgba(0,70,95,0.65)
    );
    border: 1px solid rgba(100,220,255,0.35);
    border-radius: 18px;
    padding: 24px;
    margin-top: 18px;
    margin-bottom: 22px;
}

.ai-title {
    color: #8ee7ff;
    font-size: 24px;
    font-weight: 800;
}

.ai-text {
    color: #e6f8ff;
    font-size: 16px;
    line-height: 1.7;
}

.live-box {
    background: rgba(2,25,42,0.88);
    border: 1px solid rgba(120,220,255,0.25);
    border-radius: 18px;
    padding: 20px;
    min-height: 130px;
}

.live-label {
    color: #91bacb;
    font-size: 14px;
    font-weight: 600;
}

.live-value {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.alert-critical {
    background: rgba(130,20,30,0.72);
    border: 1px solid rgba(255,100,110,0.55);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

.alert-warning {
    background: rgba(130,90,10,0.72);
    border: 1px solid rgba(255,210,80,0.55);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

.alert-normal {
    background: rgba(10,100,70,0.65);
    border: 1px solid rgba(80,240,170,0.40);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

.footer {
    text-align: center;
    color: #9ccfe0;
    padding: 35px 0 10px 0;
}

/* Sign-in / account form contrast fixes */
.stApp label, .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stForm"] label,
.stApp [data-baseweb="tab"] button,
.stApp [data-testid="stTextInput"] label {
    color: #ffffff !important;
}
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stTextInput"] textarea {
    color: #ffffff !important;
    background-color: #092b3d !important;
    caret-color: #ffffff !important;
    border: 1px solid #7dd3fc !important;
}
.stApp [data-testid="stTextInput"] input::placeholder {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}
.stApp [data-testid="stFormSubmitButton"] button {
    color: #ffffff !important;
    background: #087b83 !important;
    border: 1px solid #7dd3fc !important;
}
.stApp [data-baseweb="tab-list"] button p {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DEMO SIGN-IN / ACCOUNT ACCESS
# ============================================================

import hashlib

if "marine_users" not in st.session_state:
    # Demo account is for local project demonstration only.
    st.session_state.marine_users = {
        "operator": hashlib.sha256("marine123".encode()).hexdigest()
    }

if "marine_signed_in" not in st.session_state:
    st.session_state.marine_signed_in = False

if "marine_username" not in st.session_state:
    st.session_state.marine_username = ""

if not st.session_state.marine_signed_in:
    st.markdown(
        '<div class="main-title">🌊 Marine Energy AI Platform</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Sign in to access the monitoring dashboard</div>',
        unsafe_allow_html=True
    )
    st.write("")
    login_col1, login_col2, login_col3 = st.columns([1, 1.3, 1])
    with login_col2:
        login_tab, signup_tab = st.tabs(["🔐 Sign In", "📝 Create Account"])
        with login_tab:
            with st.form("marine_login_form"):
                login_username = st.text_input("Username", placeholder="Enter username")
                login_password = st.text_input("Password", type="password", placeholder="Enter password")
                login_submit = st.form_submit_button("Sign In", use_container_width=True)
            if login_submit:
                saved_hash = st.session_state.marine_users.get(login_username.strip())
                entered_hash = hashlib.sha256(login_password.encode()).hexdigest()
                if saved_hash and saved_hash == entered_hash:
                    st.session_state.marine_signed_in = True
                    st.session_state.marine_username = login_username.strip()
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            st.caption("Demo login: username `operator` · password `marine123`")
        with signup_tab:
            with st.form("marine_signup_form"):
                new_username = st.text_input("Choose a username", key="signup_username")
                new_password = st.text_input("Create a password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password")
                signup_submit = st.form_submit_button("Create Account", use_container_width=True)
            if signup_submit:
                clean_username = new_username.strip()
                if len(clean_username) < 3:
                    st.error("Username must contain at least 3 characters.")
                elif len(new_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif clean_username in st.session_state.marine_users:
                    st.error("That username already exists. Please choose another.")
                else:
                    st.session_state.marine_users[clean_username] = hashlib.sha256(new_password.encode()).hexdigest()
                    st.success("Account created for this running session. Please sign in from the Sign In tab.")
    st.info("This is a local demonstration sign-in. Accounts are kept in the current Streamlit session only; use a proper database and secure authentication before deploying publicly.")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "WEC_Sydney_100.csv"


@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


if not os.path.exists(DATA_FILE):

    st.error(
        "❌ WEC_Sydney_100.csv was not found. "
        "Keep the CSV file in the same folder as app.py."
    )

    st.stop()


df = load_data()


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

power_columns = [
    col for col in df.columns
    if col.startswith("Power")
]

x_columns = [
    col for col in df.columns
    if col.startswith("X")
]

y_columns = [
    col for col in df.columns
    if col.startswith("Y")
]


# ============================================================
# DATA CLEANING
# ============================================================

for col in power_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["Total_Power"] = pd.to_numeric(
    df["Total_Power"],
    errors="coerce"
)


if "qW" in df.columns:

    df["qW"] = pd.to_numeric(
        df["qW"],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["Total_Power"]
).reset_index(drop=True)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_records = len(df)

wec_count = len(power_columns)

average_power = df["Total_Power"].mean()

maximum_power = df["Total_Power"].max()

minimum_power = df["Total_Power"].min()


# ============================================================
# AI ANOMALY DETECTION
# ============================================================

ai_features = ["Total_Power"]

if "qW" in df.columns:

    ai_features.append("qW")


ai_data = df[ai_features].copy()

ai_data = ai_data.fillna(
    ai_data.median()
)


anomaly_model = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)

anomaly_model.fit(ai_data)


df["AI_Result"] = anomaly_model.predict(
    ai_data
)


df["AI_Status"] = np.where(
    df["AI_Result"] == -1,
    "Anomaly",
    "Normal"
)


anomaly_count = int(
    (df["AI_Status"] == "Anomaly").sum()
)


# ============================================================
# WEC PERFORMANCE
# ============================================================

wec_results = []


for col in power_columns:

    avg = df[col].mean()

    wec_results.append(
        {
            "WEC": col,
            "Average Power": avg
        }
    )


wec_df = pd.DataFrame(
    wec_results
)


wec_mean = wec_df["Average Power"].mean()

wec_std = wec_df["Average Power"].std()


def determine_health(value):

    if value >= wec_mean:

        return "Healthy"

    elif value >= wec_mean - wec_std:

        return "Warning"

    else:

        return "Critical"


wec_df["Health"] = (
    wec_df["Average Power"]
    .apply(determine_health)
)


healthy_count = int(
    (wec_df["Health"] == "Healthy").sum()
)

warning_count = int(
    (wec_df["Health"] == "Warning").sum()
)

critical_count = int(
    (wec_df["Health"] == "Critical").sum()
)


# ============================================================
# AI POWER FORECASTING
# ============================================================

forecast_window = 20


def create_forecast_dataset(series):

    values = np.array(series)

    X = []
    y = []

    for i in range(
        forecast_window,
        len(values)
    ):

        X.append(
            values[
                i - forecast_window:i
            ]
        )

        y.append(
            values[i]
        )

    return np.array(X), np.array(y)


forecast_values = df[
    "Total_Power"
].values


X_forecast, y_forecast = create_forecast_dataset(
    forecast_values
)


forecast_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


split_point = int(
    len(X_forecast) * 0.8
)


X_train = X_forecast[:split_point]

X_test = X_forecast[split_point:]

y_train = y_forecast[:split_point]

y_test = y_forecast[split_point:]


forecast_model.fit(
    X_train,
    y_train
)


test_predictions = forecast_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    test_predictions
)


r2 = r2_score(
    y_test,
    test_predictions
)


def generate_future_forecast(
    model,
    values,
    steps
):

    history = list(values)

    predictions = []

    for _ in range(steps):

        input_data = np.array(
            history[-forecast_window:]
        ).reshape(
            1,
            -1
        )

        prediction = model.predict(
            input_data
        )[0]

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return predictions


# ============================================================
# ADVANCED AI PERFORMANCE MODEL
# ============================================================

@st.cache_resource
def train_performance_models():

    models = {}

    metrics = {}

    for wec in power_columns:

        target = df[wec].copy()

        feature_columns = []

        # qW is used when available
        if "qW" in df.columns:

            feature_columns.append("qW")

        # Total fleet power gives contextual information
        feature_columns.append("Total_Power")

        X = df[feature_columns].copy()

        y = target.copy()

        combined = pd.concat(
            [X, y.rename("target")],
            axis=1
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(combined) < 50:

            continue

        X = combined[feature_columns]

        y = combined["target"]


        split = int(
            len(X) * 0.8
        )

        X_train = X.iloc[:split]

        X_test = X.iloc[split:]

        y_train = y.iloc[:split]

        y_test = y.iloc[split:]


        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        model_mae = mean_absolute_error(
            y_test,
            predictions
        )


        model_r2 = r2_score(
            y_test,
            predictions
        )


        models[wec] = model

        metrics[wec] = {
            "MAE": model_mae,
            "R2": model_r2
        }


    return models, metrics


performance_models, performance_metrics = (
    train_performance_models()
)


# ============================================================
# CALCULATE WEC PERFORMANCE DEVIATION
# ============================================================

performance_results = []


for wec in power_columns:

    if wec not in performance_models:

        continue


    model = performance_models[wec]


    feature_columns = []

    if "qW" in df.columns:

        feature_columns.append("qW")

    feature_columns.append(
        "Total_Power"
    )


    X_current = df[
        feature_columns
    ].copy()


    X_current = X_current.fillna(
        X_current.median()
    )


    expected = model.predict(
        X_current
    )


    actual = df[
        wec
    ].values


    mean_actual = np.mean(
        actual
    )


    mean_expected = np.mean(
        expected
    )


    deviation = (
        (mean_actual - mean_expected)
        / max(
            abs(mean_expected),
            0.000001
        )
    ) * 100


    if deviation >= -5:

        status = "Normal"

    elif deviation >= -15:

        status = "Warning"

    else:

        status = "Potential Issue"


    performance_results.append(
        {
            "WEC": wec,
            "Actual Average Power":
                mean_actual,
            "Expected Average Power":
                mean_expected,
            "Deviation (%)":
                deviation,
            "Status":
                status,
            "MAE":
                performance_metrics[wec]["MAE"],
            "R2":
                performance_metrics[wec]["R2"]
        }
    )


performance_df = pd.DataFrame(
    performance_results
)


potential_issue_count = int(
    (
        performance_df["Status"]
        == "Potential Issue"
    ).sum()
)


performance_warning_count = int(
    (
        performance_df["Status"]
        == "Warning"
    ).sum()
)


# ============================================================
# ALERT GENERATOR
# ============================================================

def generate_alerts():
    """Create alerts only for statistically unusual total-power observations."""
    alerts = []
    power = pd.to_numeric(df["Total_Power"], errors="coerce").dropna()
    if power.empty:
        return alerts
    low_critical, high_critical = power.quantile([0.01, 0.99])
    low_warning, high_warning = power.quantile([0.05, 0.95])
    for index, row in df.iterrows():
        value = pd.to_numeric(pd.Series([row.get("Total_Power")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        if value < low_critical or value > high_critical:
            severity = "Critical"
            message = "Power output is an extreme outlier compared with the dataset."
            recommendation = "Inspect this observation and verify sensor readings and operating conditions."
        elif value < low_warning or value > high_warning:
            severity = "Warning"
            message = "Power output is outside the usual operating range."
            recommendation = "Monitor this observation and compare it with nearby readings."
        else:
            continue
        alerts.append({"Record": index + 1, "Severity": severity,
                       "WEC": str(row.get("WEC", row.get("System", "System"))),
                       "Power": float(value), "Message": message,
                       "Recommendation": recommendation})
    return alerts


alerts = generate_alerts()
critical_alerts = sum(1 for a in alerts if a["Severity"] == "Critical")
warning_alerts = sum(1 for a in alerts if a["Severity"] == "Warning")
normal_observations = max(0, total_records - len(alerts))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🌊 Marine AI"
    )

    st.write(
        "Marine Energy Monitoring & AI Analytics"
    )
    st.caption(f"Signed in as: {st.session_state.marine_username}")
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.marine_signed_in = False
        st.session_state.marine_username = ""
        st.rerun()

    st.divider()


    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔄 Live Monitoring",
            "🌊 Environmental Analysis",
            "🧠 AI Performance Prediction",
            "🔮 AI Forecasting",
            "🚨 AI Alert Center",
            "⚡ Power Analytics",
            "🤖 AI Anomaly Detection",
            "🌊 WEC Performance",
            "🗺️ WEC Location Map"
        ]
    )


    st.divider()


    st.markdown(
        "### Dataset"
    )

    st.write(
        f"📊 Records: **{total_records:,}**"
    )

    st.write(
        f"🌊 WECs: **{wec_count}**"
    )

    st.write(
        "📍 Sydney WEC"
    )


    st.divider()


    st.markdown(
        "### AI Models"
    )

    st.write(
        "🧠 Isolation Forest"
    )

    st.write(
        "🌲 Random Forest"
    )

    st.write(
        "🔮 Power Forecasting"
    )

    st.write(
        "🚨 Early Warning"
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🌊 Marine Energy AI Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Wave Energy Converter Monitoring • '
    'Resource Analysis • Power Generation • '
    'AI Prediction • Anomaly Detection • '
    'Forecasting • Intelligent Alerts'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">'
        '📊 Marine Energy System Overview'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            🌊 WEC Units
            </div>

            <div class="card-value">
            {wec_count}
            </div>

            <div class="card-desc">
            Wave Energy Converters
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            📋 Dataset Records
            </div>

            <div class="card-value">
            {total_records:,}
            </div>

            <div class="card-desc">
            Marine observations
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            ⚡ Average Power
            </div>

            <div class="card-value">
            {average_power:,.0f}
            </div>

            <div class="card-desc">
            Overall generation
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            🚨 Active Alerts
            </div>

            <div class="card-value">
            {critical_alerts + warning_alerts}
            </div>

            <div class="card-desc">
            AI-generated alerts
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">'
        '⚡ Total Power Generation'
        '</div>',
        unsafe_allow_html=True
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(
                1,
                len(df) + 1
            ),
            y=df["Total_Power"],
            mode="lines",
            name="Total Power"
        )
    )


    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="Marine Energy Power Output",
        xaxis_title="Observation",
        yaxis_title="Total Power"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.markdown(
        '<div class="section-title">📊 Power Distribution</div>',
        unsafe_allow_html=True
    )
    dist_fig = px.histogram(
        df, x="Total_Power", nbins=25,
        title="Distribution of Power Output",
        labels={"Total_Power": "Total Power"},
        template="plotly_dark"
    )
    dist_fig.update_layout(height=330, showlegend=False)
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">'
        '🩺 Fleet Health Summary'
        '</div>',
        unsafe_allow_html=True
    )


    h1, h2, h3 = st.columns(3)


    with h1:

        st.success(
            f"🟢 Healthy WECs: {healthy_count}"
        )


    with h2:

        st.warning(
            f"🟡 Warning WECs: {warning_count}"
        )


    with h3:

        st.error(
            f"🔴 Critical WECs: {critical_count}"
        )


    st.markdown(
        f"""
        <div class="ai-box">

        <div class="ai-title">
        🤖 AI Platform Summary
        </div>

        <div class="ai-text">

        The system analysed
        <b>{total_records:,}</b> observations
        across <b>{wec_count}</b> Wave Energy Converter units.

        <br><br>

        AI anomaly detection identified
        <b>{anomaly_count}</b> unusual observations.

        <br><br>

        The forecasting model achieved an R² score of
        <b>{r2:.3f}</b>.

        <br><br>

        The performance model identified
        <b>{potential_issue_count}</b> potential performance
        issues and <b>{performance_warning_count}</b> warnings.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "🔄 Live Monitoring":

    st.markdown(
        '<div class="section-title">'
        '🔄 Live Marine Energy Monitoring'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-box">

        <div class="ai-title">
        🌊 Simulated Real-Time Monitoring
        </div>

        <div class="ai-text">

        Historical WEC observations are replayed
        sequentially to demonstrate continuous monitoring.

        The AI system evaluates every incoming observation.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        start_monitoring = st.button(
            "▶️ Start Monitoring",
            width="stretch"
        )


    with col2:

        stop_monitoring = st.button(
            "⏹️ Stop",
            width="stretch"
        )


    with col3:

        speed = st.selectbox(
            "Monitoring Speed",
            [
                "Fast",
                "Normal",
                "Slow"
            ]
        )


    if "live_index" not in st.session_state:

        st.session_state.live_index = 0


    if "monitoring" not in st.session_state:

        st.session_state.monitoring = False


    if start_monitoring:

        st.session_state.monitoring = True


    if stop_monitoring:

        st.session_state.monitoring = False


    if speed == "Fast":

        delay = 0.2

    elif speed == "Normal":

        delay = 0.7

    else:

        delay = 1.5


    index = st.session_state.live_index

    current = df.iloc[index]


    current_power = float(
        current["Total_Power"]
    )


    current_qw = 0

    if "qW" in df.columns:

        current_qw = float(
            current["qW"]
        )


    current_input = pd.DataFrame(
        [[
            current_power,
            current_qw
        ]],
        columns=ai_features
    )


    current_ai = anomaly_model.predict(
        current_input
    )[0]


    if current_ai == -1:

        status = "🔴 ANOMALY"

    else:

        status = "🟢 NORMAL"


    st.markdown(
        "### 📡 Current Sensor Reading"
    )


    l1, l2, l3, l4 = st.columns(4)


    with l1:

        st.markdown(
            f"""
            <div class="live-box">

            <div class="live-label">
            ⚡ TOTAL POWER
            </div>

            <div class="live-value">
            {current_power:,.0f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with l2:

        st.markdown(
            f"""
            <div class="live-box">

            <div class="live-label">
            🌊 qW
            </div>

            <div class="live-value">
            {current_qw:.3f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with l3:

        st.markdown(
            f"""
            <div class="live-box">

            <div class="live-label">
            📍 RECORD
            </div>

            <div class="live-value">
            {index + 1}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with l4:

        st.markdown(
            f"""
            <div class="live-box">

            <div class="live-label">
            🤖 AI STATUS
            </div>

            <div class="live-value">
            {status}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    live_data = df[
        "Total_Power"
    ].iloc[
        max(0, index - 50):index + 1
    ]


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(
                max(0, index - 50) + 1,
                index + 2
            ),
            y=live_data,
            mode="lines+markers",
            name="Power"
        )
    )


    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Incoming Power Stream",
        xaxis_title="Record",
        yaxis_title="Power"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.progress(
        (index + 1) / total_records
    )


    st.write(
        f"Monitoring record **{index + 1} / {total_records}**"
    )


    if st.session_state.monitoring:

        time.sleep(delay)

        if index < total_records - 1:

            st.session_state.live_index += 1

        else:

            st.session_state.live_index = 0

        st.rerun()


# ============================================================
# ENVIRONMENTAL ANALYSIS
# ============================================================

elif page == "🌊 Environmental Analysis":

    st.markdown(
        '<div class="section-title">'
        '🌊 Environmental & Resource Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-box">

        <div class="ai-title">
        🌊 Marine Resource → Energy Analysis
        </div>

        <div class="ai-text">

        This module studies the relationship between
        <b>qW</b>, the resource-related variable available
        in this dataset, and WEC power generation.

        <br><br>

        The system does not assume that qW is directly
        equal to wave height, tidal speed, or current speed.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if "qW" not in df.columns:

        st.warning(
            "The dataset does not contain a qW column."
        )

    else:

        qw_mean = df["qW"].mean()

        qw_max = df["qW"].max()

        qw_min = df["qW"].min()

        qw_std = df["qW"].std()


        r1, r2, r3, r4 = st.columns(4)


        with r1:

            st.metric(
                "Average qW",
                f"{qw_mean:.3f}"
            )


        with r2:

            st.metric(
                "Maximum qW",
                f"{qw_max:.3f}"
            )


        with r3:

            st.metric(
                "Minimum qW",
                f"{qw_min:.3f}"
            )


        with r4:

            st.metric(
                "qW Variation",
                f"{qw_std:.3f}"
            )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=np.arange(
                    1,
                    len(df) + 1
                ),
                y=df["qW"],
                mode="lines",
                name="qW"
            )
        )


        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="qW Resource Indicator",
            xaxis_title="Observation",
            yaxis_title="qW"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        scatter = px.scatter(
            df,
            x="qW",
            y="Total_Power",
            title="qW vs Total Power"
        )


        scatter.update_layout(
            template="plotly_dark",
            height=500
        )


        st.plotly_chart(
            scatter,
            width="stretch"
        )


        correlation = df[
            ["qW", "Total_Power"]
        ].corr().iloc[0, 1]


        st.metric(
            "qW ↔ Total Power Correlation",
            f"{correlation:.3f}"
        )


        if correlation >= 0.7:

            interpretation = (
                "Strong positive relationship observed."
            )

        elif correlation >= 0.3:

            interpretation = (
                "Moderate positive relationship observed."
            )

        elif correlation > -0.3:

            interpretation = (
                "Weak linear relationship observed."
            )

        elif correlation > -0.7:

            interpretation = (
                "Moderate negative relationship observed."
            )

        else:

            interpretation = (
                "Strong negative relationship observed."
            )


        st.info(
            interpretation
        )


        try:

            df["qW_Level"] = pd.qcut(
                df["qW"],
                5,
                labels=[
                    "Very Low",
                    "Low",
                    "Medium",
                    "High",
                    "Very High"
                ],
                duplicates="drop"
            )


            summary = (
                df.groupby(
                    "qW_Level",
                    observed=True
                )["Total_Power"]
                .agg(
                    [
                        "mean",
                        "min",
                        "max",
                        "count"
                    ]
                )
                .reset_index()
            )


            summary.columns = [
                "qW Condition",
                "Average Power",
                "Minimum Power",
                "Maximum Power",
                "Observations"
            ]


            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )


            bar = px.bar(
                summary,
                x="qW Condition",
                y="Average Power",
                title="Average Power by qW Condition"
            )


            bar.update_layout(
                template="plotly_dark",
                height=430
            )


            st.plotly_chart(
                bar,
                width="stretch"
            )

        except Exception:

            st.warning(
                "Unable to create qW condition groups."
            )


        st.markdown(
            f"""
            <div class="ai-box">

            <div class="ai-title">
            🤖 AI Resource Interpretation
            </div>

            <div class="ai-text">

            {interpretation}

            <br><br>

            The result describes a statistical relationship
            in the available dataset. Correlation alone does
            not prove physical causation.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 🧠 AI PERFORMANCE PREDICTION
# ============================================================

elif page == "🧠 AI Performance Prediction":

    st.markdown(
        '<div class="section-title">'
        '🧠 AI Performance Prediction & Early Warning'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-box">

        <div class="ai-title">
        🤖 WEC Performance Intelligence
        </div>

        <div class="ai-text">

        The AI model learns the relationship between the
        available resource/system variables and each WEC's
        historical power output.

        <br><br>

        It then compares <b>actual power</b> with
        <b>model-expected power</b>.

        <br><br>

        A large persistent deviation is flagged as a
        <b>potential performance issue</b> for further
        investigation.

        <br><br>

        ⚠️ This is an AI early-warning system, not a
        confirmed physical fault diagnosis.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "WECs Analysed",
            len(performance_df)
        )


    with m2:

        st.metric(
            "Potential Issues",
            potential_issue_count
        )


    with m3:

        st.metric(
            "Warnings",
            performance_warning_count
        )


    with m4:

        average_r2 = performance_df[
            "R2"
        ].mean()

        st.metric(
            "Average Model R²",
            f"{average_r2:.3f}"
        )


    # --------------------------------------------------------
    # STATUS DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        "### 🩺 AI Performance Status"
    )


    status_counts = (
        performance_df["Status"]
        .value_counts()
        .reset_index()
    )


    status_counts.columns = [
        "Status",
        "Count"
    ]


    status_fig = px.bar(
        status_counts,
        x="Status",
        y="Count",
        title="WEC Performance Status"
    )


    status_fig.update_layout(
        template="plotly_dark",
        height=400
    )


    st.plotly_chart(
        status_fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # DEVIATION CHART
    # --------------------------------------------------------

    st.markdown(
        "### 📉 Actual vs Expected Performance"
    )


    deviation_df = performance_df.sort_values(
        "Deviation (%)"
    )


    deviation_fig = px.bar(
        deviation_df,
        x="WEC",
        y="Deviation (%)",
        color="Status",
        title="WEC Performance Deviation"
    )


    deviation_fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="WEC",
        yaxis_title="Deviation (%)"
    )


    st.plotly_chart(
        deviation_fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # WEC SELECTOR
    # --------------------------------------------------------

    st.markdown(
        "### 🔍 Detailed WEC Analysis"
    )


    selected_wec = st.selectbox(
        "Select a WEC",
        list(performance_df["WEC"])
    )


    selected_result = performance_df[
        performance_df["WEC"] == selected_wec
    ].iloc[0]


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "Actual Average",
            f"{selected_result['Actual Average Power']:,.2f}"
        )


    with s2:

        st.metric(
            "Expected Average",
            f"{selected_result['Expected Average Power']:,.2f}"
        )


    with s3:

        st.metric(
            "Deviation",
            f"{selected_result['Deviation (%)']:.2f}%"
        )


    with s4:

        st.metric(
            "Model R²",
            f"{selected_result['R2']:.3f}"
        )


    # --------------------------------------------------------
    # ACTUAL VS EXPECTED GRAPH
    # --------------------------------------------------------

    model = performance_models[
        selected_wec
    ]


    features = []

    if "qW" in df.columns:

        features.append("qW")

    features.append(
        "Total_Power"
    )


    selected_input = df[
        features
    ].copy()


    selected_input = selected_input.fillna(
        selected_input.median()
    )


    expected_series = model.predict(
        selected_input
    )


    actual_series = df[
        selected_wec
    ].fillna(
        df[selected_wec].median()
    ).values


    comparison_count = min(
        250,
        len(actual_series)
    )


    actual_recent = actual_series[
        -comparison_count:
    ]

    expected_recent = expected_series[
        -comparison_count:
    ]


    comparison_fig = go.Figure()


    comparison_fig.add_trace(
        go.Scatter(
            x=np.arange(
                1,
                comparison_count + 1
            ),
            y=actual_recent,
            mode="lines",
            name="Actual Power"
        )
    )


    comparison_fig.add_trace(
        go.Scatter(
            x=np.arange(
                1,
                comparison_count + 1
            ),
            y=expected_recent,
            mode="lines",
            name="AI Expected Power"
        )
    )


    comparison_fig.update_layout(
        template="plotly_dark",
        height=500,
        title=f"{selected_wec}: Actual vs AI Expected Power",
        xaxis_title="Recent Observation",
        yaxis_title="Power"
    )


    st.plotly_chart(
        comparison_fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # AI RECOMMENDATION
    # --------------------------------------------------------

    deviation = selected_result[
        "Deviation (%)"
    ]


    if deviation < -15:

        recommendation = (
            f"🔴 {selected_wec} shows a significant "
            f"negative performance deviation of "
            f"{abs(deviation):.1f}%. Further investigation "
            "is recommended. Compare environmental/resource "
            "conditions and equipment operating data before "
            "confirming any physical fault."
        )

        st.error(
            recommendation
        )


    elif deviation < -5:

        recommendation = (
            f"🟡 {selected_wec} is performing below the "
            f"model expectation by {abs(deviation):.1f}%. "
            "Continue monitoring for a persistent trend."
        )

        st.warning(
            recommendation
        )


    else:

        recommendation = (
            f"🟢 {selected_wec} is operating close to "
            "its model expectation based on the available data."
        )

        st.success(
            recommendation
        )


    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Complete AI Performance Report"
    )


    formatted_performance = performance_df.copy()


    st.dataframe(
        formatted_performance,
        width="stretch",
        hide_index=True
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 AI Feature Importance"
    )


    importance = model.feature_importances_


    importance_df = pd.DataFrame(
        {
            "Feature":
                features,

            "Importance":
                importance
        }
    ).sort_values(
        "Importance",
        ascending=False
    )


    importance_fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title=f"Features Influencing {selected_wec} Prediction"
    )


    importance_fig.update_layout(
        template="plotly_dark",
        height=400
    )


    st.plotly_chart(
        importance_fig,
        width="stretch"
    )


# ============================================================
# AI FORECASTING
# ============================================================

elif page == "🔮 AI Forecasting":

    st.markdown(
        '<div class="section-title">'
        '🔮 AI Power Forecasting'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-box">

        <div class="ai-title">
        🌲 Random Forest Power Forecast
        </div>

        <div class="ai-text">

        The model learns historical Total Power patterns
        and predicts upcoming power observations using
        the previous 20 observations.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    f1, f2, f3 = st.columns(3)


    with f1:

        st.metric(
            "MAE",
            f"{mae:,.2f}"
        )


    with f2:

        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )


    with f3:

        st.metric(
            "Training Records",
            f"{len(X_train):,}"
        )


    forecast_steps = st.slider(
        "Future observations",
        5,
        50,
        15
    )


    future_predictions = generate_future_forecast(
        forecast_model,
        forecast_values,
        forecast_steps
    )


    recent_count = min(
        100,
        len(forecast_values)
    )


    recent_actual = forecast_values[
        -recent_count:
    ]


    actual_x = np.arange(
        1,
        len(recent_actual) + 1
    )


    future_x = np.arange(
        len(recent_actual) + 1,
        len(recent_actual) +
        forecast_steps + 1
    )


    forecast_fig = go.Figure()


    forecast_fig.add_trace(
        go.Scatter(
            x=actual_x,
            y=recent_actual,
            mode="lines",
            name="Historical Power"
        )
    )


    forecast_fig.add_trace(
        go.Scatter(
            x=future_x,
            y=future_predictions,
            mode="lines+markers",
            name="AI Forecast"
        )
    )


    forecast_fig.update_layout(
        template="plotly_dark",
        height=500,
        title="Historical Power vs AI Forecast",
        xaxis_title="Observation",
        yaxis_title="Power"
    )


    st.plotly_chart(
        forecast_fig,
        width="stretch"
    )


    predicted_average = np.mean(
        future_predictions
    )

    predicted_max = np.max(
        future_predictions
    )

    predicted_min = np.min(
        future_predictions
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "Predicted Average",
            f"{predicted_average:,.0f}"
        )


    with p2:

        st.metric(
            "Predicted Maximum",
            f"{predicted_max:,.0f}"
        )


    with p3:

        st.metric(
            "Predicted Minimum",
            f"{predicted_min:,.0f}"
        )


    forecast_table = pd.DataFrame(
        {
            "Future Observation":
                range(
                    1,
                    forecast_steps + 1
                ),

            "Predicted Power":
                future_predictions
        }
    )


    st.dataframe(
        forecast_table,
        width="stretch",
        hide_index=True
    )


# ============================================================
# AI ALERT CENTER
# ============================================================

elif page == "🚨 AI Alert Center":

    st.markdown(
        '<div class="section-title">'
        '🚨 AI Alert & Recommendation Center'
        '</div>',
        unsafe_allow_html=True
    )


    a1, a2, a3, a4 = st.columns(4)


    with a1:

        st.metric(
            "Total Alerts",
            len(alerts)
        )


    with a2:

        st.metric(
            "Critical",
            critical_alerts
        )


    with a3:

        st.metric(
            "Warnings",
            warning_alerts
        )


    with a4:

        st.metric(
            "Normal Records",
            normal_observations
        )

    st.caption("Critical = extreme power outlier (below 1st or above 99th percentile). Warning = unusual power (below 5th or above 95th percentile). Other records are treated as normal.")


    alert_filter = st.selectbox(
        "Filter Severity",
        [
            "All",
            "Critical",
            "Warning"
        ]
    )


    filtered = alerts


    if alert_filter != "All":

        filtered = [
            a
            for a in alerts
            if a["Severity"]
            == alert_filter
        ]


    if len(filtered) == 0:

        st.success(
            f"🟢 Normal operation for this filter. {normal_observations:,} records are within the expected power range."
        )

    else:

        for alert in filtered[:40]:

            if alert["Severity"] == "Critical":

                st.markdown(
                    f"""
                    <div class="alert-critical">

                    <b>🔴 CRITICAL • {alert["WEC"]}</b>

                    <br><br>

                    Record: {alert["Record"]}

                    <br>

                    Power:
                    {alert["Power"]:,.2f}

                    <br><br>

                    <b>AI Finding:</b>
                    {alert["Message"]}

                    <br><br>

                    <b>Recommendation:</b>
                    {alert["Recommendation"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="alert-warning">

                    <b>🟡 WARNING • {alert["WEC"]}</b>

                    <br><br>

                    Record: {alert["Record"]}

                    <br>

                    Power:
                    {alert["Power"]:,.2f}

                    <br><br>

                    <b>AI Finding:</b>
                    {alert["Message"]}

                    <br><br>

                    <b>Recommendation:</b>
                    {alert["Recommendation"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    if alerts:

        st.markdown(
            "### 📋 Alert History"
        )


        alert_df = pd.DataFrame(
            alerts
        )


        st.dataframe(
            alert_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# POWER ANALYTICS
# ============================================================

elif page == "⚡ Power Analytics":

    st.markdown(
        '<div class="section-title">'
        '⚡ Power Generation Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    fig1 = px.histogram(
        df,
        x="Total_Power",
        nbins=40,
        title="Total Power Distribution"
    )


    fig1.update_layout(
        template="plotly_dark",
        height=430
    )


    st.plotly_chart(
        fig1,
        width="stretch"
    )


    rolling = df[
        "Total_Power"
    ].rolling(
        30
    ).mean()


    fig2 = go.Figure()


    fig2.add_trace(
        go.Scatter(
            y=df["Total_Power"],
            mode="lines",
            name="Power"
        )
    )


    fig2.add_trace(
        go.Scatter(
            y=rolling,
            mode="lines",
            name="30 Record Average"
        )
    )


    fig2.update_layout(
        template="plotly_dark",
        height=430,
        title="Power Trend with Moving Average",
        xaxis_title="Observation",
        yaxis_title="Power"
    )


    st.plotly_chart(
        fig2,
        width="stretch"
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🤖 AI Anomaly Detection":

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Anomaly Detection'
        '</div>',
        unsafe_allow_html=True
    )


    a1, a2, a3 = st.columns(3)


    with a1:

        st.metric(
            "Records",
            f"{total_records:,}"
        )


    with a2:

        st.metric(
            "Normal",
            f"{total_records - anomaly_count:,}"
        )


    with a3:

        st.metric(
            "Anomalies",
            f"{anomaly_count:,}"
        )


    chart_df = df.reset_index()


    fig = px.scatter(
        chart_df,
        x="index",
        y="Total_Power",
        color="AI_Status",
        title="AI-Detected Power Anomalies"
    )


    fig.update_layout(
        template="plotly_dark",
        height=500
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    anomaly_df = df[
        df["AI_Status"] == "Anomaly"
    ]


    st.dataframe(
        anomaly_df[
            ["Total_Power"] +
            (
                ["qW"]
                if "qW" in df.columns
                else []
            ) +
            ["AI_Status"]
        ].head(100),
        width="stretch"
    )


# ============================================================
# WEC PERFORMANCE
# ============================================================

elif page == "🌊 WEC Performance":

    st.markdown(
        '<div class="section-title">'
        '🌊 WEC Performance Monitoring'
        '</div>',
        unsafe_allow_html=True
    )


    top = wec_df.sort_values(
        "Average Power",
        ascending=False
    ).head(20)


    fig = px.bar(
        top,
        x="WEC",
        y="Average Power",
        title="WEC Average Power Comparison"
    )


    fig.update_layout(
        template="plotly_dark",
        height=500
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    selected = st.selectbox(
        "Select WEC",
        power_columns
    )


    selected_values = df[
        selected
    ]


    w1, w2, w3 = st.columns(3)


    with w1:

        st.metric(
            "Average Power",
            f"{selected_values.mean():,.0f}"
        )


    with w2:

        st.metric(
            "Maximum Power",
            f"{selected_values.max():,.0f}"
        )


    with w3:

        st.metric(
            "Minimum Power",
            f"{selected_values.min():,.0f}"
        )


    individual_df = pd.DataFrame(
        {
            "Observation":
                range(
                    1,
                    len(selected_values) + 1
                ),

            "Power":
                selected_values.values
        }
    )


    fig2 = px.line(
        individual_df,
        x="Observation",
        y="Power",
        title=f"{selected} Power Generation"
    )


    fig2.update_layout(
        template="plotly_dark",
        height=450
    )


    st.plotly_chart(
        fig2,
        width="stretch"
    )


    st.dataframe(
        wec_df.sort_values(
            "Average Power",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )


# ============================================================
# LOCATION MAP - ENERGY POTENTIAL INDICATORS
# ============================================================

elif page == "🗺️ WEC Location Map":

    st.markdown(
        '<div class="section-title">🗺️ India Marine Energy Potential Map</div>',
        unsafe_allow_html=True
    )
    st.write("Hover over or select a coastal marker to inspect its relative wave and tidal energy potential.")
    st.warning(
        "Important: your CSV contains X/Y dataset coordinates, not Indian latitude/longitude, and it does not include clearly named wave-speed or tidal/current-speed columns. "
        "So the India coast pins below are illustrative demo locations. Low/Medium/High values are preliminary power-proxy categories from your available Power columns, not live or site-specific measurements."
    )

    # Find actual power fields in the dataset and calculate a dataset-based proxy.
    power_cols = [c for c in df.columns if str(c).lower().startswith("power")]
    if not power_cols and "Total_Power" in df.columns:
        power_cols = ["Total_Power"]
    power_values = []
    for col in power_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not vals.empty:
            power_values.extend(vals.tolist())
    if not power_values:
        st.error("No usable Power columns were found in WEC_Sydney_100.csv, so potential levels cannot be estimated.")
    else:
        power_values = np.asarray(power_values, dtype=float)
        low_cut = float(np.quantile(power_values, 1/3))
        high_cut = float(np.quantile(power_values, 2/3))

        def power_level(value):
            if value >= high_cut:
                return "High"
            if value >= low_cut:
                return "Medium"
            return "Low"

        # Approximate coastal demonstration pins around India. These are not dataset GPS coordinates.
        coastal_sites = [
            {"Site": "Gujarat Coast", "lat": 22.30, "lon": 69.00},
            {"Site": "Mumbai Coast", "lat": 19.00, "lon": 72.80},
            {"Site": "Goa Coast", "lat": 15.30, "lon": 73.80},
            {"Site": "Karnataka Coast", "lat": 14.80, "lon": 74.10},
            {"Site": "Kerala Coast", "lat": 9.50, "lon": 76.20},
            {"Site": "Kanyakumari", "lat": 8.08, "lon": 77.55},
            {"Site": "Chennai Coast", "lat": 13.08, "lon": 80.30},
            {"Site": "Puducherry Coast", "lat": 11.93, "lon": 79.83},
            {"Site": "Andhra Coast", "lat": 16.50, "lon": 82.00},
            {"Site": "Odisha Coast", "lat": 19.80, "lon": 85.80},
            {"Site": "West Bengal Coast", "lat": 21.60, "lon": 88.10},
            {"Site": "Andaman & Nicobar", "lat": 11.70, "lon": 92.70},
            {"Site": "Lakshadweep", "lat": 10.60, "lon": 72.60},
        ]
        site_rows = []
        for i, site in enumerate(coastal_sites):
            # Use available dataset power values as a transparent demo proxy for the site marker.
            sample_power = float(np.quantile(power_values, (i + 1) / (len(coastal_sites) + 1)))
            level = power_level(sample_power)
            site_rows.append({
                **site,
                "Estimated Power (dataset units)": sample_power,
                "Wave Potential (proxy)": level,
                "Tidal Potential (proxy)": level,
                "Potential": level,
            })
        india_df = pd.DataFrame(site_rows)
        palette = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}
        # Use scatter_geo rather than scatter_mapbox for compatibility across
        # Plotly versions. It renders an India-focused geographic map without
        # requiring the Mapbox-specific API.
        fig = px.scatter_geo(
            india_df,
            lat="lat", lon="lon", color="Potential", size="Estimated Power (dataset units)",
            size_max=22,
            hover_name="Site",
            hover_data={
                "Wave Potential (proxy)": True,
                "Tidal Potential (proxy)": True,
                "Estimated Power (dataset units)": ":.3f",
                "lat": False, "lon": False, "Potential": False,
            },
            color_discrete_map=palette,
            title="India Coast — Illustrative Marine Energy Potential"
        )
        fig.update_traces(marker={"opacity": 0.92, "line": {"width": 1, "color": "white"}})
        fig.update_geos(
            scope="asia",
            projection_type="natural earth",
            center={"lat": 20.5, "lon": 81.0},
            lataxis_range=[-2, 36],
            lonaxis_range=[60, 101],
            showcountries=True,
            countrycolor="#64748b",
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            showland=True,
            landcolor="#263445",
            showocean=True,
            oceancolor="#0b2239",
            bgcolor="#101827",
        )
        fig.update_layout(
            template="plotly_dark", height=650,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            legend_title="Relative potential",
            paper_bgcolor="#101827",
        )
        st.plotly_chart(fig, use_container_width=True, key="india_marine_energy_map")

        st.markdown("### 🎨 Potential legend")
        l1, l2, l3 = st.columns(3)
        l1.success("🟢 HIGH — upper third of available power values")
        l2.warning("🟠 MEDIUM — middle third of available power values")
        l3.error("🔴 LOW — lower third of available power values")

        st.markdown("### 📍 Select a coastal spot")
        selected_site_name = st.selectbox("Choose a marker / coastal location", india_df["Site"].tolist())
        selected_site = india_df.loc[india_df["Site"] == selected_site_name].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Wave Power Potential (proxy)", selected_site["Wave Potential (proxy)"])
        b.metric("Tidal Power Potential (proxy)", selected_site["Tidal Potential (proxy)"])
        c.metric("Estimated Power (dataset units)", f'{selected_site["Estimated Power (dataset units)"]:,.3f}')
        st.caption("The wave and tidal labels currently share the same power proxy because the CSV has no separate wave/tidal measurement columns. Add real site latitude/longitude plus wave and current/tidal measurements to make this scientifically site-specific.")
        st.dataframe(
            india_df[["Site", "Estimated Power (dataset units)", "Wave Potential (proxy)", "Tidal Potential (proxy)"]],
            hide_index=True, use_container_width=True
        )
