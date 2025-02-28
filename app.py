import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # Layout
        with st.container():

            with col1:
                st.subheader("Original Text")
                st.write(raw_text)

                st.subheader("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.markdown(f"**Prediction:**  {prediction} {emoji_icon}")
                st.markdown(f"**Accuracy:**  {np.max(probability):.2f}")

            with col2:
                st.subheader("Prediction Probability")

                # Convert probability array to DataFrame
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                # Create an Altair bar chart
                chart = (
                    alt.Chart(proba_df_clean)
                    .mark_bar()
                    .encode(
                        x=alt.X("Emotions", sort="-y"),
                        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                        color="Emotions"
                    )
                    .properties(width=400, height=300)
                )
                st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()