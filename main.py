import nltk
import pickle
import re
import streamlit as st
from io import StringIO
from pdfminer.high_level import extract_text
import time
from streamlit_lottie import st_lottie
import json
import requests
from streamlit.components.v1 import html

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
@st.cache_resource
def load_models():
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    return clf, tfidf

clf, tfidf = load_models()

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation URLs
lottie_upload = "https://assets8.lottiefiles.com/packages/lf20_vnikrcia.json"
lottie_typing = "https://assets1.lottiefiles.com/packages/lf20_4kx2q32h.json"

# Clean resume text
def cleanResume(txt):
    cleantxt = re.sub('http\S+\s', ' ', txt)
    cleantxt = re.sub('@\S+', ' ', cleantxt)
    cleantxt = re.sub('#\S+', " ", cleantxt)
    cleantxt = re.sub('RT|CC', ' ', cleantxt)
    cleantxt = re.sub(r'[!"#&%\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleantxt)
    cleantxt = re.sub('[^a-zA-Z]', ' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    cleantxt = cleantxt.lower()
    return cleantxt

# Extract text from PDF using pdfminer
def extract_text_from_pdf(file):
    return extract_text(file)

# Web App
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background-color: #f4f6f9;
        }

        .title-text {
            color: #0e1117;
            text-align: center;
            font-size: 3em;
            margin-bottom: 0.5em;
            font-weight: 700;
        }
        .subheader {
            color: #4b4f56;
            text-align: center;
            font-size: 1.4em;
            margin-bottom: 2em;
        }
        .upload-section {
            border: 2px dashed #6c63ff;
            background-color: #ffffff;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 30px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            color: #999;
            font-size: 0.9em;
            margin-top: 4em;
            padding: 2em 0;
            border-top: 1px solid #eaeaea;
        }
        .result-box {
            background-color: #ffffff;
            color: #0e1117;
            padding: 25px;
            border-radius: 10px;
            font-size: 1.8em;
            font-weight: bold;
            margin: 40px auto;
            max-width: 600px;
            text-align: center;
            border: 2px solid #6c63ff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title-text">🚀 Resume Screening Pro</div>', unsafe_allow_html=True)

    lottie_typing_json = load_lottieurl(lottie_typing)
    if lottie_typing_json:
        st_lottie(lottie_typing_json, height=120, key="typing", speed=1)

    st.markdown('<div class="subheader">Upload your resume in PDF or TXT format for instant analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader('', type=['txt', 'pdf'], label_visibility="collapsed")

    if uploaded_file is not None:
        lottie_upload_json = load_lottieurl(lottie_upload)
        if lottie_upload_json:
            st_lottie(lottie_upload_json, height=120, key="upload", speed=1)

    if uploaded_file is not None:
        try:
            with st.spinner('🔍 Analyzing your resume...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.015)
                    progress_bar.progress(percent_complete + 1)

                if uploaded_file.type == "text/plain":
                    resume_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                elif uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return

                cleaned_resume = cleanResume(resume_text)
                input_feature = tfidf.transform([cleaned_resume])
                prediction_id = clf.predict(input_feature)[0]

                category_mapping = {
                    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
                    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science",
                    7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
                    10: "ETL Developer", 11: "Electrical Engineering", 12: "HR",
                    13: "Hadoop", 14: "Health and fitness", 15: "Java Developer",
                    16: "Mechanical Engineer", 17: "Network Security Engineer",
                    18: "Operations Manager", 19: "PMO", 20: "Python Developer",
                    21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
                }

                category_name = category_mapping.get(prediction_id, "Unknown")

                progress_bar.empty()
                st.markdown(f"""
                <div class="result-box">
                    🎯 Predicted Job Category: <br> <span style='color:#6c63ff'>{category_name}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please try uploading a different file.")

    st.markdown("""
    <div class="footer">
        <p>🚀 Powered by AI and Streamlit | Developed to revolutionize resume screening</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
