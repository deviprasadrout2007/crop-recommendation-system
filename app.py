import io
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
)

FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
DISPLAY_NAMES = {
    "n": "Nitrogen (N)",
    "p": "Phosphorus (P)",
    "k": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}
DATASET_URL = (
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/"
    "Data-processed/crop_recommendation.csv"
)


@st.cache_data
def load_from_url():
    with urllib.request.urlopen(DATASET_URL, timeout=20) as response:
        return pd.read_csv(io.BytesIO(response.read()))


@st.cache_data
def read_uploaded_file(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def clean_dataset(data):
    data = data.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]

    if "label" in data.columns:
        target = "label"
    elif "crop" in data.columns:
        target = "crop"
    else:
        raise ValueError("Dataset must contain a 'label' or 'crop' column.")

    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data[FEATURES + [target]].copy()
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[target] = data[target].astype(str).str.strip()
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna(subset=FEATURES + [target])
    data = data[data[target] != ""].drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid records remain after cleaning.")

    if data[target].nunique() < 2:
        raise ValueError("At least two crop classes are required.")

    if data[target].value_counts().min() < 2:
        raise ValueError("Every crop class needs at least two records.")

    return data, target


def create_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf")),
            ]
        ),
    }


@st.cache_resource
def train_models(X_train, y_train):
    models = create_models()
    trained = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        rows.append({"Model": name, "Training Accuracy (%)": round(
            accuracy_score(y_train, predictions) * 100, 2
        )})
        trained[name] = model

    return trained, pd.DataFrame(rows)


st.title("🌱 Crop Recommendation System")
st.write(
    "A machine-learning application that recommends a crop using "
    "soil nutrients and climate conditions."
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a compatible CSV file", type=["csv"])
    st.caption("Required: n, p, k, temperature, humidity, ph, rainfall, label")

try:
    if uploaded is not None:
        raw_df = read_uploaded_file(uploaded.getvalue())
        source_message = "Uploaded dataset"
    else:
        raw_df = load_from_url()
        source_message = "Dataset loaded from the public URL"

    df, target = clean_dataset(raw_df)
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.info("Upload a compatible Crop_recommendation.csv file from the sidebar.")
    st.stop()

st.success(source_message)

with st.expander("Dataset overview", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Crop Classes", df[target].nunique())
    st.dataframe(df.head(10), use_container_width=True)

st.header("📈 Exploratory Data Analysis")
tab1, tab2, tab3 = st.tabs(["Crop Distribution", "Feature Distribution", "Correlation"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=target, order=sorted(df[target].unique()), ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Crop Class Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    chosen_feature = st.selectbox(
        "Select a feature",
        FEATURES,
        format_func=lambda x: DISPLAY_NAMES[x],
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x=target, y=chosen_feature, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"{DISPLAY_NAMES[chosen_feature]} by Crop")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

X = df[FEATURES]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

trained_models, training_table = train_models(X_train, y_train)

st.header("🤖 Model Comparison")
test_rows = []
for name, model in trained_models.items():
    predictions = model.predict(X_test)
    test_rows.append({
        "Model": name,
        "Test Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
    })
test_table = pd.DataFrame(test_rows)
st.dataframe(test_table, use_container_width=True)

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=test_table, x="Model", y="Test Accuracy (%)", ax=ax)
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=20)
ax.set_title("Test Accuracy Comparison")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("🔄 Stratified Cross-Validation")
folds = min(5, int(y_train.value_counts().min()))
if folds >= 2:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_rows = []
    for name in trained_models:
        model = create_models()[name]
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_rows.append({
            "Model": name,
            "Mean CV Accuracy (%)": round(scores.mean() * 100, 2),
            "CV Std Dev (%)": round(scores.std() * 100, 2),
        })
    cv_table = pd.DataFrame(cv_rows)
    st.dataframe(cv_table, use_container_width=True)
else:
    st.warning("Not enough samples per class for cross-validation.")

st.header("🔍 Detailed Evaluation")
default_model = test_table.loc[test_table["Test Accuracy (%)"].idxmax(), "Model"]
selected_name = st.selectbox(
    "Choose a model",
    list(trained_models.keys()),
    index=list(trained_models.keys()).index(default_model),
)
selected_model = trained_models[selected_name]
predictions = selected_model.predict(X_test)

st.subheader("Classification Report")
report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.subheader("Confusion Matrix")
labels = sorted(y.unique())
matrix = confusion_matrix(y_test, predictions, labels=labels)
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted Crop")
ax.set_ylabel("Actual Crop")
ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("📌 Feature Importance")
if selected_name in ["Decision Tree", "Random Forest"]:
    native = pd.Series(
        selected_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({
            "Feature": [DISPLAY_NAMES[x] for x in native.index],
            "Importance": native.values,
        }).round(4),
        use_container_width=True,
    )

try:
    permutation = permutation_importance(
        selected_model, X_test, y_test, n_repeats=5,
        random_state=42, scoring="accuracy", n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "Feature": [DISPLAY_NAMES[x] for x in FEATURES],
        "Mean Importance": permutation.importances_mean,
        "Std Dev": permutation.importances_std,
    }).sort_values("Mean Importance", ascending=False)
    st.dataframe(perm_df.round(4), use_container_width=True)
except Exception as error:
    st.warning(f"Permutation importance unavailable: {error}")

st.header("🌾 Crop Prediction")
st.caption("Enter values in the units used by your dataset.")

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        n_value = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, 1.0)
        p_value = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0, 1.0)
        k_value = st.number_input("Potassium (K)", 0.0, 250.0, 43.0, 1.0)
        temperature_value = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, 0.1)
    with right:
        humidity_value = st.number_input("Humidity (%)", 0.0, 100.0, 80.0, 0.1)
        ph_value = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
        rainfall_value = st.number_input("Rainfall (mm)", 0.0, 5000.0, 200.0, 1.0)
    submitted = st.form_submit_button("🌱 Recommend Crop")

if submitted:
    input_df = pd.DataFrame([{
        "n": n_value,
        "p": p_value,
        "k": k_value,
        "temperature": temperature_value,
        "humidity": humidity_value,
        "ph": ph_value,
        "rainfall": rainfall_value,
    }])
    crop = selected_model.predict(input_df)[0]
    st.success(f"Recommended Crop: {str(crop).upper()}")
    st.info(
        "This is a model prediction, not a guarantee. "
        "Consult local agricultural guidance before planting."
    )

st.markdown("---")
st.caption("B.Tech CSE AI/ML Project | Crop Recommendation System")
