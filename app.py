#!/usr/bin/env python

import os
import base64
import io
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ==============================
# DASH APP INITIALIZATION
# ==============================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

MASTER_DF = None


# ==============================
# PAGE 1 – DATA UPLOAD
# ==============================

def page1_layout():

    return dbc.Container([

        html.H2("Upload Dataset"),

        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
            style={
                "width": "100%",
                "height": "60px",
                "borderStyle": "dashed",
                "textAlign": "center"
            }
        ),

        html.Br(),

        html.Div(id="upload-status"),

        html.Hr(),

        html.Div(id="data-preview")

    ])


# ==============================
# PAGE 2 – EDA
# ==============================

def page2_layout():

    return dbc.Container([

        html.H2("Exploratory Data Analysis"),

        dcc.Dropdown(
            id="corr-cols",
            multi=True,
            placeholder="Select columns for correlation"
        ),

        dbc.Button("Generate Heatmap", id="corr-btn"),

        html.Br(),
        html.Br(),

        html.Div(id="corr-out")

    ])


# ==============================
# PAGE 3 – ML MODELS
# ==============================

def page3_layout():

    return dbc.Container([

        html.H2("Machine Learning Models"),

        dcc.Dropdown(
            id="model-name",
            options=[
                {"label": "Logistic Regression", "value": "blr"},
                {"label": "Decision Tree", "value": "dt"},
                {"label": "Random Forest", "value": "rf"},
                {"label": "SVM", "value": "svm"},
                {"label": "Naive Bayes", "value": "nb"},
                {"label": "KNN", "value": "knn"}
            ],
            placeholder="Select ML model"
        ),

        dbc.Button("Train Model", id="train-btn"),

        html.Br(),
        html.Br(),

        html.Div(id="model-output")

    ])


# ==============================
# PAGE 4 – SENTIMENT ANALYSIS
# ==============================

def page4_layout():

    return dbc.Container([

        html.H2("Sentiment Analysis"),

        dcc.Upload(
            id="upload-sentiment",
            children=html.Div(["Upload News CSV"]),
            style={
                "width": "100%",
                "height": "60px",
                "borderStyle": "dashed",
                "textAlign": "center"
            }
        ),

        html.Br(),

        html.Div(id="sentiment-output")

    ])


# ==============================
# MAIN LAYOUT
# ==============================

app.layout = dbc.Container([

    html.H1("Stock Market ML Dashboard"),

    dbc.Tabs(
        id="tabs",
        active_tab="page1",
        children=[

            dbc.Tab(label="Upload Data", tab_id="page1"),
            dbc.Tab(label="EDA", tab_id="page2"),
            dbc.Tab(label="ML Models", tab_id="page3"),
            dbc.Tab(label="Sentiment", tab_id="page4")

        ]
    ),

    html.Div(id="tab-content")

])


# ==============================
# TAB ROUTING
# ==============================

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):

    if tab == "page1":
        return page1_layout()

    if tab == "page2":
        return page2_layout()

    if tab == "page3":
        return page3_layout()

    if tab == "page4":
        return page4_layout()


# ==============================
# DATA UPLOAD CALLBACK
# ==============================

@app.callback(
    [Output("upload-status", "children"),
     Output("data-preview", "children")],

    Input("upload-data", "contents"),
    State("upload-data", "filename"),

    prevent_initial_call=True
)
def load_data(contents, filename):

    global MASTER_DF

    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    MASTER_DF = df.copy()

    status = dbc.Alert(
        f"{filename} loaded successfully | Rows: {df.shape[0]}",
        color="success"
    )

    preview = dash_table.DataTable(
        data=df.head(10).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns]
    )

    return status, preview


# ==============================
# CORRELATION HEATMAP
# ==============================

@app.callback(
    Output("corr-out", "children"),
    Input("corr-btn", "n_clicks"),
    State("corr-cols", "value"),
    prevent_initial_call=True
)
def corr(_, cols):

    if MASTER_DF is None:
        return "Upload dataset first"

    fig = px.imshow(MASTER_DF[cols].corr(), text_auto=True)

    return dcc.Graph(figure=fig)


# ==============================
# TRAIN MODEL
# ==============================

@app.callback(
    Output("model-output", "children"),
    Input("train-btn", "n_clicks"),
    State("model-name", "value"),
    prevent_initial_call=True
)
def train_model(_, model_name):

    if MASTER_DF is None:
        return "Upload dataset first"

    df = MASTER_DF.dropna()

    X = df.drop("Nifty_Open_Dir", axis=1)
    y = df["Nifty_Open_Dir"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    models = {

        "blr": LogisticRegression(max_iter=2000),
        "dt": DecisionTreeClassifier(),
        "rf": RandomForestClassifier(),
        "svm": SVC(probability=True),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=7)

    }

    model = models[model_name]

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr))

    fig.update_layout(title=f"ROC Curve | AUC = {roc_auc:.3f}")

    return dcc.Graph(figure=fig)


# ==============================
# SENTIMENT ANALYSIS
# ==============================

@app.callback(
    Output("sentiment-output", "children"),
    Input("upload-sentiment", "contents"),
    prevent_initial_call=True
)
def sentiment(contents):

    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    text = " ".join(df["text"].dropna())

    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)

    plt.axis("off")

    buf = io.BytesIO()

    plt.savefig(buf, format="png")

    buf.seek(0)

    img = base64.b64encode(buf.read()).decode()

    return html.Img(src="data:image/png;base64," + img)


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8050))

    app.run(host="0.0.0.0", port=port)
