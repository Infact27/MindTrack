import streamlit as st 
import joblib
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# Load models and thresholds
@st.cache_resource      # Cache the loaded models to avoid reloading on every interaction
def load_assets():
    emotion_model = joblib.load("app/emotion_model.joblib")
    stress_model = joblib.load("app/stress_model.joblib")
    phq9_model = joblib.load("app/phq9_model.joblib")
    mlb = joblib.load("app/emotion_mlb.joblib")

    with open("app/emotion_thresholds.json", "r") as f:
        thresholds = np.array(json.load(f))

    return emotion_model, stress_model, phq9_model, mlb, thresholds

emotion_model, stress_model, phq9_model, mlb, thresholds = load_assets()

# Load emotion labels
@st.cache_resource   # Cache the loaded labels 
def load_emotion_labels():
    with open("app/emotions.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

emotion_labels_list = load_emotion_labels()

# Text preprocessing utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def remove_custom_stopword(text):
    custom_stopwords = stop_words.union({
    'lol', 'thanks', 'youre', 'im', 'dont', 'get', 'name'
    })
    tokens = text.split()
    tokens = [t for t in tokens if t not in custom_stopwords]
    return " ".join(tokens)


# PHQ-9 interpretation
def interpret_phq9(score):
    score = int(round(score))

    if 0 <= score <= 4:
        severity = "None-minimal"
        action = (
            "No immediate treatment may be necessary. "
            "Maintaining healthy routines and self-monitoring is recommended."
        )
    elif 5 <= score <= 9:
        severity = "Mild"
        action = (
            "Consider monitoring symptoms and discussing concerns with a "
            "healthcare professional if symptoms persist."
        )
    elif 10 <= score <= 14:
        severity = "Moderate"
        action = (
            "Consultation with a healthcare professional is recommended to "
            "evaluate symptoms and possible treatment options."
        )
    elif 15 <= score <= 19:
        severity = "Moderately severe"
        action = (
            "Professional evaluation is strongly recommended. "
            "Treatment options may include psychotherapy, medication, or both."
        )
    else:
        severity = "Severe"
        action = (
            "Immediate professional support is strongly recommended. "
            "Please consider contacting a mental health professional or "
            "local support services."
        )

    return score, severity, action

# Severity color mapping
def severity_color(severity):
    return {
        "None-minimal": "green",
        "Mild": "blue",
        "Moderate": "orange",
        "Moderately severe": "darkorange",
        "Severe": "red"
    }.get(severity, "black")


# Streamlit App
st.set_page_config(page_title="MindTrack Dashboard")

st.title("MindTrack NLP Dashboard")
st.write(
    "This dashboard demonstrates emotion, stress level, "
    "and PHQ-9 score prediction from journal text."
)

# user input
st.header("Journal Input")

user_text = st.text_area(
    "Write your daily journal:",
    height=200,
    placeholder="Today I felt anxious and tired about my future..."
)


# Predict
if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_text)
        cleaned_text = remove_custom_stopword(cleaned_text)

        # emotion prediction
        emotion_probs = emotion_model.predict_proba([cleaned_text])[0]
        emotion_binary = (emotion_probs >= thresholds).astype(int)
        emotion_binary_2d = np.array([emotion_binary])
        emotion_labels = mlb.inverse_transform(emotion_binary_2d)[0]
        emotion_names = [emotion_labels_list[i] for i in emotion_labels]
        
        # stress prediction
        stress_pred = stress_model.predict([cleaned_text])[0]
        stress_label = "Stressed" if stress_pred == 1 else "Not Stressed"

        # PHQ-9 prediction
        phq9_pred = phq9_model.predict([cleaned_text])[0]
        phq9_score, phq9_severity, phq9_action = interpret_phq9(phq9_pred)
        # ======================
        # Output
        # ======================
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Emotion(s)**")
            if emotion_names:
                for e in emotion_names:
                    st.write(f"- {e.capitalize()}")
            else:
                st.write("No dominant emotion detected")

        with col2:
            st.metric("Stress Level", stress_label)

        with col3:
            st.metric("PHQ-9 Score", phq9_score)
            color = severity_color(phq9_severity)
            st.markdown(
                f"**Severity:** <span style='color:{color}'>{phq9_severity}</span>",
                unsafe_allow_html=True
            )
            st.caption(
                    "PHQ-9 is a standardized questionnaire used to screen depressive symptoms."
                )

        with st.expander("What is the PHQ-9 score?"):
            st.markdown(
                """
                **PHQ-9 (Patient Health Questionnaire-9)** is a clinically validated
                screening tool used to assess the presence and severity of depressive
                symptoms.

                **How it works**
                - Consists of 9 questions related to mood, energy, sleep, appetite,
                concentration, and interest in daily activities
                - Each item is scored from 0 to 3
                - Total score ranges from **0 to 27**
                - Higher scores indicate more severe depressive symptoms

                **How PHQ-9 is different from stress**
                - **Stress** reflects short-term psychological pressure or tension
                - **PHQ-9** reflects depressive symptoms experienced over a longer
                period and their impact on daily functioning
                - A person may experience stress without depression, or depression
                without feeling acute stress

                **Important note**
                This tool is intended for screening purposes only and does not replace
                professional diagnosis.
                """
            )

        st.subheader("Suggested Action (Based on PHQ-9 Guideline)")

        st.info(phq9_action)

        st.caption(
            "⚠️ This output is generated by a machine learning model and "
            "PHQ-9 guidelines. It is not a medical diagnosis."
        )
    
