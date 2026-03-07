#!/usr/bin/env python
# coding: utf-8

# In[6]:





# In[7]:





# In[8]:


import vaderSentiment
print("VADER Installed Successfully!")


# In[9]:





# In[10]:





# In[12]:


 # ===============================
# IMPORTS
# ===============================
import base64
import io
import pandas as pd
import numpy as np
import scipy.stats as stats

from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

# ===============================
# APP INITIALIZATION
# ===============================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# ===============================
# GLOBAL DATA HOLDER
# ===============================
MASTER_DF = None

# ===============================
# PAGE 1 LAYOUT
# ===============================
def page1_layout():
    return dbc.Container([

        html.H2("📁 Page 1 – Upload Dataset", className="mt-4 mb-4"),

        dcc.Upload(
            id="upload-data",
            children=html.Div([
                "Drag & Drop or ",
                html.A("Select CSV File")
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "20px"
            },
            multiple=False
        ),

        html.Div(id="upload-status"),
        html.Hr(),

        html.H5("Data Preview (first 20 rows)"),
        html.Div(id="data-preview")

    ], fluid=True)

# ===============================
# PAGE 2 LAYOUT – ADVANCED EDA
# ===============================
def page2_layout():
    return dbc.Container([

        html.H2("📊 Page 2 – Exploratory Data Analysis", className="mb-4"),

        # 1️⃣ YEAR-WISE DISTRIBUTION
        dbc.Card([
            dbc.CardHeader("1️⃣ Year-wise Distribution of Index Returns"),
            dbc.CardBody([
                dcc.Dropdown(id="box-cols", multi=True),
                dbc.Button("Generate Box Plot", id="box-btn", className="mt-2"),
                html.Div(id="box-out")
            ])
        ], className="mb-4"),

        # 2️⃣ CORRELATION HEATMAP
        dbc.Card([
            dbc.CardHeader("2️⃣ Correlation Heatmap of Market Returns"),
            dbc.CardBody([
                dcc.Dropdown(id="corr-cols", multi=True),
                dbc.Button("Generate Heatmap", id="corr-btn", className="mt-2"),
                html.Div(id="corr-out")
            ])
        ], className="mb-4"),

        # 3️⃣ ROLLING RISK ANALYSIS
        dbc.Card([
            dbc.CardHeader("3️⃣ Rolling Mean & Volatility"),
            dbc.CardBody([
                dcc.Dropdown(id="roll-col"),
                dcc.Slider(
                    id="roll-window",
                    min=10, max=60, step=10, value=20,
                    marks={i: str(i) for i in range(10, 70, 10)}
                ),
                dbc.Button("Generate Rolling Analysis", id="roll-btn", className="mt-3"),
                html.Div(id="roll-out")
            ])
        ], className="mb-4"),

        # 4️⃣ DISTRIBUTION vs NORMALITY
        dbc.Card([
            dbc.CardHeader("4️⃣ Return Distribution vs Normal Distribution"),
            dbc.CardBody([
                dcc.Dropdown(id="dist-col"),
                dbc.Button("Generate Distribution Plot", id="dist-btn", className="mt-2"),
                html.Div(id="dist-out")
            ])
        ], className="mb-4"),

        # 5️⃣ TARGET-WISE FEATURE ANALYSIS
        dbc.Card([
            dbc.CardHeader("5️⃣ Feature Behaviour by Nifty Opening Direction"),
            dbc.CardBody([
                dcc.Dropdown(id="target-features", multi=True),
                dcc.Dropdown(
                    id="stat-type",
                    options=[
                        {"label": "Mean", "value": "mean"},
                        {"label": "Median", "value": "median"},
                        {"label": "Standard Deviation", "value": "std"}
                    ],
                    value="mean"
                ),
                dbc.Button("Generate Target-wise Analysis", id="target-btn", className="mt-2"),
                html.Div(id="target-out")
            ])
        ],className="mb-4"),

        # 6️⃣ FEATURE IMPORTANCE
        dbc.Card([
            dbc.CardHeader("6️⃣ Feature Importance (Top Predictors)"),
            dbc.CardBody([
                dbc.Button(
                    "Generate Feature Importance",
                    id="featimp-btn",
                    className="mt-2"
                ),
                html.Div(id="featimp-out")
            ])
        ])

    ], fluid=True)


# ===============================
# MAIN LAYOUT WITH TABS
# ===============================
app.layout = dbc.Container([

    html.H1("📊 Stock Market ML Dashboard", className="text-center mt-3"),

    dbc.Tabs(
        id="tabs",
        active_tab="page1",
        children=[
            dbc.Tab(label="📁 Page 1: Data Upload", tab_id="page1"),
            dbc.Tab(label="📊 Page 2: EDA", tab_id="page2"),
            dbc.Tab(label="📈 Page 3: ML Models", tab_id="page3"),
            dbc.Tab(label="📰 Page 4: Sentiment", tab_id="page4"),

        ]
    ),

    html.Div(id="tab-content", className="mt-4")

], fluid=True)


# ===============================
# TAB ROUTING CALLBACK
# ===============================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):

    if tab == "page1":
        return dbc.Container([
            html.H2("📁 Page 1 – Upload Dataset", className="mt-4 mb-4"),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag & Drop or ",
                    html.A("Select CSV File")
                ]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "marginBottom": "20px"
                },
                multiple=False
            ),
            html.Div(id="upload-status"),
            html.Hr(),
            html.H5("Data Preview (first 20 rows)"),
            html.Div(id="data-preview")
        ], fluid=True)

    elif tab == "page2":
        return page2_layout()

    elif tab == "page3":
        return page3_layout()

    elif tab == "page4":
        return page4_layout()



# ===============================
# PAGE 1 CALLBACK – LOAD CSV
# ===============================
@app.callback(
    [Output("upload-status", "children"),
     Output("data-preview", "children")],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def load_user_data(contents, filename):
    global MASTER_DF

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    MASTER_DF = df.copy()

    status = dbc.Alert(
        f"✅ {filename} loaded | Rows: {df.shape[0]} | Columns: {df.shape[1]}",
        color="success"
    )

    preview = dash_table.DataTable(
        data=df.head(20).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 12, "textAlign": "left"}
    )

    return status, preview

# ===============================
# POPULATE RETURN COLUMNS
# ===============================
@app.callback(
    [
        Output("box-cols", "options"),
        Output("corr-cols", "options"),
        Output("roll-col", "options"),
        Output("dist-col", "options"),
        Output("target-features", "options")
    ],
    Input("tabs", "value")
)
def populate_columns(_):
    if MASTER_DF is None:
        return [], [], [], [], []

    return_cols = [c for c in MASTER_DF.columns if c.endswith("_Return")]
    opts = [{"label": c, "value": c} for c in return_cols]
    return opts, opts, opts, opts, opts

# ===============================
# EDA CALLBACKS
# ===============================
@app.callback(
    Output("box-out", "children"),
    Input("box-btn", "n_clicks"),
    State("box-cols", "value"),
    prevent_initial_call=True
)
def boxplot(_, cols):

    # 🔐 SAFETY CHECK
    if cols is None or len(cols) == 0:
        return dbc.Alert("⚠️ Please select at least one column.", color="warning")

    graphs = []

    for c in cols:
        fig = px.box(
            MASTER_DF,
            x="Year",
            y=c,
            title=f"Year-wise Distribution of {c}"
        )
        fig.update_traces(boxmean=True)

        graphs.append(
            dcc.Graph(
                figure=fig
            )
        )

    return graphs

@app.callback(Output("corr-out", "children"),
              Input("corr-btn", "n_clicks"),
              State("corr-cols", "value"),
              prevent_initial_call=True)
def corr(_, cols):
    fig = px.imshow(MASTER_DF[cols].corr(), text_auto=".2f", title="Correlation Heatmap")
    return dcc.Graph(figure=fig)

@app.callback(Output("roll-out", "children"),
              Input("roll-btn", "n_clicks"),
              State("roll-col", "value"),
              State("roll-window", "value"),
              prevent_initial_call=True)
def rolling(_, col, window):
    df = MASTER_DF.copy()
    df["Rolling Mean"] = df[col].rolling(window).mean()
    df["Rolling Volatility"] = df[col].rolling(window).std()
    fig = px.line(df, y=["Rolling Mean", "Rolling Volatility"])
    return dcc.Graph(figure=fig)

@app.callback(Output("dist-out", "children"),
              Input("dist-btn", "n_clicks"),
              State("dist-col", "value"),
              prevent_initial_call=True)
def dist(_, col):
    data = MASTER_DF[col].dropna()
    fig = px.histogram(data, nbins=60, histnorm="probability density")
    x = np.linspace(data.min(), data.max(), 200)
    fig.add_scatter(x=x, y=stats.norm.pdf(x, data.mean(), data.std()),
                    mode="lines", name="Normal")
    return dcc.Graph(figure=fig)

@app.callback(Output("target-out", "children"),
              Input("target-btn", "n_clicks"),
              State("target-features", "value"),
              State("stat-type", "value"),
              prevent_initial_call=True)
def target(_, features, stat):
    grouped = MASTER_DF.groupby("Nifty_Open_Dir")[features].agg(stat)
    fig = px.bar(grouped, barmode="group")
    return dcc.Graph(figure=fig)

@app.callback(
    Output("featimp-out", "children"),
    Input("featimp-btn", "n_clicks"),
    prevent_initial_call=True
)
def feature_importance_plot(n):

    if MASTER_DF is None:
        return dbc.Alert("⚠️ Dataset not loaded yet.", color="danger")

    df = MASTER_DF.dropna()

    # ✅ ONLY RETURN COLUMNS
    return_features = [c for c in df.columns if c.endswith("_Return")]

    if len(return_features) == 0:
        return dbc.Alert("⚠️ No Return columns found!", color="warning")

    X = df[return_features]
    y = df["Nifty_Open_Dir"]

    # ✅ Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)

    imp = pd.Series(rf.feature_importances_, index=return_features)
    imp = imp.sort_values(ascending=False).head(15)

    # ✅ Plot
    fig = px.bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        title="Top Return Features Influencing Nifty Opening Direction",
        labels={"x": "Importance Score", "y": "Return Feature"}
    )

    return dcc.Graph(figure=fig)


# ============================================================
# ✅ PAGE 3 – FINAL MODEL TRAINING + COMPARISON DASHBOARD
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# =======================
# ✅ PAGE 3 LAYOUT
# =======================
def page3_layout():
    return dbc.Container([

        html.H2("📈 Page 3 – ML Model Training & Comparison", className="mb-4"),

        dbc.Row([

            dbc.Col([
                html.Label("📅 Select Training Date Range"),
                dcc.DatePickerRange(
                    id="train-range",
                    display_format="YYYY-MM-DD"
                )
            ], md=4),

            dbc.Col([
                html.Label("🧪 Test Period (Last Months)"),
                dcc.Dropdown(
                    id="test-months",
                    options=[
                        {"label": "Last 3 Months", "value": 3},
                        {"label": "Last 6 Months", "value": 6},
                        {"label": "Last 12 Months", "value": 12},
                    ],
                    value=6
                )
            ], md=3),

            dbc.Col([
                html.Label("📌 Select Return Features"),
                dcc.Dropdown(
                    id="feature-cols",
                    multi=True
                )
            ], md=5),

        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("⚙️ Select Model"),
                dcc.Dropdown(
                    id="model-name",
                    options=[
                        {"label": "Binary Logistic Regression", "value": "blr"},
                        {"label": "Decision Tree", "value": "dt"},
                        {"label": "Random Forest", "value": "rf"},
                        {"label": "Support Vector Machine", "value": "svm"},
                        {"label": "Naive Bayes", "value": "nb"}
                    ]
                )
            ], md=4),
        ], className="mb-3"),

        dbc.Button("🚀 Train Selected Model", id="train-btn", color="primary"),
        dbc.Button("📊 Compare All Models", id="compare-btn", color="dark", className="ms-3"),

        html.Hr(),

        html.Div(id="model-output"),
        html.Div(id="compare-output")

    ], fluid=True)



# =======================
# ✅ Populate Return Features Only
# =======================
@app.callback(
    Output("feature-cols", "options"),
    Input("tabs", "active_tab")
)
def populate_return_features(tab):

    if MASTER_DF is None:
        return []

    return_cols = [c for c in MASTER_DF.columns if c.endswith("_Return")]

    return [{"label": c, "value": c} for c in return_cols]



# =======================
# ✅ Fix Training Date Range Calendar
# =======================
@app.callback(
    Output("train-range", "min_date_allowed"),
    Output("train-range", "max_date_allowed"),
    Output("train-range", "start_date"),
    Output("train-range", "end_date"),
    Input("tabs", "active_tab")
)
def set_calendar_range(tab):

    if MASTER_DF is None:
        return None, None, None, None

    df = MASTER_DF.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    return min_date, max_date, min_date, max_date



# =======================
# ✅ TRAIN SELECTED MODEL
# =======================
@app.callback(
    Output("model-output", "children"),
    Input("train-btn", "n_clicks"),
    State("train-range", "start_date"),
    State("train-range", "end_date"),
    State("test-months", "value"),
    State("feature-cols", "value"),
    State("model-name", "value"),
    prevent_initial_call=True
)
def train_model(n, start, end, test_months, features, model_name):

    if features is None or model_name is None:
        return dbc.Alert("⚠️ Please select Features + Model first.", color="warning")

    df = MASTER_DF.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ✅ Apply Training Range Filter
    train_df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    # ✅ Test = Last N months
    cutoff = train_df["Date"].max() - pd.DateOffset(months=test_months)

    test_df = train_df[train_df["Date"] >= cutoff]
    train_df = train_df[train_df["Date"] < cutoff]

    # ✅ Drop NaN
    train_df = train_df.dropna(subset=features + ["Nifty_Open_Dir"])
    test_df = test_df.dropna(subset=features + ["Nifty_Open_Dir"])

    X_train = train_df[features]
    y_train = train_df["Nifty_Open_Dir"]

    X_test = test_df[features]
    y_test = test_df["Nifty_Open_Dir"]

    # =======================
    # ✅ Models
    # =======================
    models = {
        "blr": LogisticRegression(max_iter=2000),
        "dt": DecisionTreeClassifier(max_depth=5),
        "rf": RandomForestClassifier(n_estimators=200),
        "svm": SVC(probability=True),
        "nb": GaussianNB()
    }

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", models[model_name])
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # ================= ROC + AUC =================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    auc_card = dbc.Alert(f"✅ AUC Score = {roc_auc:.3f}", color="success")

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines"))
    roc_fig.update_layout(title="ROC Curve")

    # ================= Confusion Matrix =================
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")

    # ================= Tree Plot =================
    tree_plot = None
    if model_name == "dt":

        dt_model = pipe.named_steps["model"]

        plt.figure(figsize=(12,6))
        plot_tree(dt_model, feature_names=features, filled=True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode()

        tree_plot = html.Img(
            src="data:image/png;base64," + encoded,
            style={"width": "100%", "marginTop": "20px"}
        )

    return [
        auc_card,
        dcc.Graph(figure=roc_fig),
        dcc.Graph(figure=cm_fig),
        tree_plot
    ]



# =======================
# ✅ COMPARE ALL MODELS
# =======================
@app.callback(
    Output("compare-output", "children"),
    Input("compare-btn", "n_clicks"),
    State("feature-cols", "value"),
    prevent_initial_call=True
)
def compare_models(n, features):

    if features is None:
        return dbc.Alert("⚠️ Please select features first!", color="warning")

    df = MASTER_DF.copy()
    df = df.dropna(subset=features + ["Nifty_Open_Dir"])

    X = df[features]
    y = df["Nifty_Open_Dir"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "BLR": LogisticRegression(max_iter=2000),
        "DT": DecisionTreeClassifier(max_depth=5),
        "RF": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True),
        "NB": GaussianNB()
    }

    results = []

    for name, model in models.items():

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc_score = auc(*roc_curve(y_test, y_prob)[:2])

        results.append({"Model": name, "AUC": round(auc_score, 3)})

    res_df = pd.DataFrame(results)

    bar_fig = px.bar(res_df, x="Model", y="AUC", title="Model Comparison (AUC Scores)")

    return [
        dash_table.DataTable(
            data=res_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in res_df.columns],
            style_cell={"textAlign": "center"}
        ),
        dcc.Graph(figure=bar_fig)
    ]

# ============================================================
# PAGE 4 – SENTIMENT ANALYSIS (UPLOAD + WORDCLOUD + BASIC NLP)
# ============================================================

import base64
import io
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64

# ============================================================
# PAGE 4 LAYOUT
# ============================================================

def page4_layout():
    return dbc.Container([

        html.H2("📰 Page 4 – Sentiment Analysis Dashboard", className="mb-4"),

        # Upload CSV
        dcc.Upload(
            id="upload-sentiment",
            children=html.Div([
                "Drag & Drop or CSV Upload"
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "textAlign": "center"
            },
            multiple=False
        ),

        html.Br(),
        html.Div(id="sentiment-status"),

        html.Hr(),

        # Outputs
        html.Div(id="wordcloud-output"),
        html.Br(),
        html.Div(id="sentiment-chart")

    ], fluid=True)


# ============================================================
# CALLBACK – LOAD DATA + ANALYSIS
# ============================================================

@app.callback(
    [
        Output("sentiment-status", "children"),
        Output("wordcloud-output", "children"),
        Output("sentiment-chart", "children")
    ],
    Input("upload-sentiment", "contents"),
    State("upload-sentiment", "filename"),
    prevent_initial_call=True
)
def run_sentiment(contents, filename):

    # ----------------------------
    # Load CSV
    # ----------------------------
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    # ----------------------------
    # Ensure text column exists
    # ----------------------------
    if "text" not in df.columns:
        return (
            dbc.Alert("❌ CSV must contain a 'text' column.", color="danger"),
            None,
            None
        )

    # ----------------------------
    # Clean HTML tags
    # ----------------------------
    df["text"] = df["text"].astype(str).str.replace("<.*?>", "", regex=True)

    # Combine all text
    all_text = " ".join(df["text"].dropna())

    # ----------------------------
    # WORDCLOUD
    # ----------------------------
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(all_text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    encoded_img = base64.b64encode(buf.read()).decode()

    wordcloud_img = html.Img(
        src="data:image/png;base64," + encoded_img,
        style={"width": "100%", "border": "1px solid black"}
    )

    # ----------------------------
    # SIMPLE SENTIMENT ANALYSIS
    # ----------------------------
    positive_words = ["gain","up","positive","bull","rise","profit"]
    negative_words = ["loss","down","negative","bear","fall","crash"]

    def simple_sentiment(text):
        text = text.lower()

        pos = sum(w in text for w in positive_words)
        neg = sum(w in text for w in negative_words)

        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["text"].apply(simple_sentiment)

    # ----------------------------
    # Sentiment Chart
    # ----------------------------
    fig = px.histogram(
        df,
        x="Sentiment",
        title="Sentiment Distribution of News Articles"
    )

    return (
        dbc.Alert(f"✅ {filename} Uploaded Successfully!", color="success"),
        wordcloud_img,
        dcc.Graph(figure=fig)
    )

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)

# In[ ]:





# In[ ]:





# In[ ]:




