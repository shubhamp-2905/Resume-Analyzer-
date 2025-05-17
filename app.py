import nltk
import pickle
import re
import streamlit as st
from io import StringIO
from pdfminer.high_level import extract_text

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

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
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                # Read text file directly
                resume_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            elif uploaded_file.type == "application/pdf":
                # Extract text from PDF
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
            st.success(f"Predicted Resume Category: **{category_name}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
