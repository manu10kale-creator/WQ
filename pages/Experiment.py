import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, GridSearchCV


if all(key not in st.session_state.keys() for key in ('model','num_features', 'score')):
    st.session_state['model']= []
    st.session_state['num_features']= []
    st.session_state['score']= []

def train_models():
    df = pd.read_csv("winequality-red.csv")
    X = df.drop('quality',axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

    results = []

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    results.append(("Baseline", X_train.shape[1], f1_score(y_test, y_pred_dummy, average="macro")))

    # Decision Tree
    dt_params = {"max_depth": [3,5], "min_samples_leaf":[1,2]}
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=3, scoring="f1_macro", n_jobs=1)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results.append(("Decision Tree", X_train.shape[1], f1_score(y_test, y_pred_dt, average="macro")))

    # Random Forest
    rf_params = {"n_estimators":[100], "max_depth":[None,10], "min_samples_leaf":[1]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), rf_params, cv=3, scoring="f1_macro", n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(("Random Forest", X_train.shape[1], f1_score(y_test, y_pred_rf, average="macro")))

    # Gradient Boosting
    gb_params = {"n_estimators":[100], "learning_rate":[0.1], "max_depth":[3]}
    gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=3, scoring="f1_macro", n_jobs=1)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results.append(("Gradient Boosting", X_train.shape[1], f1_score(y_test, y_pred_gb, average="macro")))

    # Update session_state
    for model, num_feat, f1 in results:
        st.session_state['model'].append(model)
        st.session_state['num_features'].append(num_feat)
        st.session_state['score'].append(round(f1, 3))

    # Confusion Matrices
    models = {
        "Baseline": y_pred_dummy,
        "Decision Tree": y_pred_dt,
        "Random Forest": y_pred_rf,
        "Gradient Boosting": y_pred_gb
    }
    for name, y_pred in models.items():
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
        disp.plot(ax=ax, xticks_rotation="vertical")
        plt.title(f"{name} — Confusion Matrix")
        st.pyplot(fig)

# Page content
st.title("⚙️ Train Models")

if st.button("🚀 Run Training"):
    train_models()
    st.success("✅ Training complete! Go back to '🏆 Model ranking' to compare results.")

