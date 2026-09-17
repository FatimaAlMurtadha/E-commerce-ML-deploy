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