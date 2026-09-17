import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC