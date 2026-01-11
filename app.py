import streamlit as st
import PyPDF2
import docx2txt
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ==============================
# NLTK Setup
# ==============================
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ==============================
# Job Roles & Skills
# ==============================
job_roles = {
    "Data Analyst": ["excel", "sql", "python", "data visualization", "pandas", "tableau", "power bi"],
    "Web Developer": ["html", "css", "javascript", "react", "wordpress", "php", "nodejs"],
    "AI/ML Engineer": ["python", "tensorflow", "pytorch", "machine learning", "deep learning", "numpy", "pandas"],
    "Cybersecurity Analyst": ["network security", "firewall", "ethical hacking", "penetration testing", "cryptography", "vulnerability analysis"],
    "Software Engineer": ["java", "c++", "python", "data structures", "algorithms", "git", "problem solving"],
    "Cloud Engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "linux", "terraform"],
    "DevOps Engineer": ["ci/cd", "jenkins", "docker", "kubernetes", "linux", "cloud infrastructure"],
    "UI/UX Designer": ["figma", "adobe xd", "wireframing", "prototyping", "user research", "design thinking"],
    "Content Writer": ["seo", "blog writing", "copywriting", "editing", "social media content", "wordpress"],
    "Data Scientist": ["python", "statistics", "machine learning", "data visualization", "pandas", "numpy", "matplotlib"],
    "Mobile App Developer": ["flutter", "dart", "java", "kotlin", "android studio", "firebase"],
    "Blockchain Developer": ["solidity", "ethereum", "smart contracts", "web3", "cryptography", "decentralized apps"]
}

# ==============================
# Improvement Tips
# ==============================
role_tips = {
    "Data Analyst": "Learn Power BI/Tableau, practice SQL queries, and strengthen Python & Excel data analysis.",
    "Web Developer": "Master HTML/CSS, learn React or Node.js, and build real-world websites or WordPress themes.",
    "AI/ML Engineer": "Practice Python, study ML algorithms, learn TensorFlow/PyTorch, and build AI-based projects.",
    "Cybersecurity Analyst": "Improve penetration testing, ethical hacking, and network monitoring skills with practical labs.",
    "Software Engineer": "Focus on DSA, problem-solving on LeetCode, and learn OOPs, Git, and version control best practices.",
    "Cloud Engineer": "Get familiar with AWS/Azure, learn Docker & Kubernetes, and practice deploying cloud applications.",
    "DevOps Engineer": "Understand CI/CD pipelines, Jenkins, Docker, and Kubernetes. Learn cloud automation tools like Terraform.",
    "UI/UX Designer": "Enhance your Figma or Adobe XD skills, practice wireframing & prototyping, and study user behavior.",
    "Content Writer": "Improve SEO writing, content research, and storytelling; build a writing portfolio on LinkedIn or Medium.",
    "Data Scientist": "Study statistics, build ML models, and improve data visualization and storytelling with data.",
    "Mobile App Developer": "Learn Flutter or Kotlin, focus on UI/UX design for mobile, and build your own mini apps for GitHub.",
    "Blockchain Developer": "Master Solidity, learn Ethereum smart contracts, and understand cryptographic principles and Web3."
}

# ==============================
# Text Preprocessing
# ==============================
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ==============================
# Resume Reader
# ==============================
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    else:
        return ""

# ==============================
# Extract Candidate Name
# ==============================
def extract_candidate_name(text, filename):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        if 2 <= len(line.split()) <= 4 and not line.lower() == "resume":
            return line
    return filename.split(".")[0]

# ==============================
# Suggest Job Roles
# ==============================
def suggest_job_role(resume_text, top_n=3):
    resume_text = preprocess_text(resume_text)
    role_scores = {}
    for role, skills in job_roles.items():
        skills_text = " ".join(skills)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([skills_text, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        role_scores[role] = score

    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_roles[:top_n]

# ==============================
# Check Perfect Skill Match
# ==============================
def is_perfect_match(resume_text, role):
    resume_text = preprocess_text(resume_text)
    role_skills = [skill.lower() for skill in job_roles[role]]
    return all(skill in resume_text for skill in role_skills)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Career Guidance Tool", page_icon="🎯", layout="wide")
st.title("🎯 Career Guidance Tool for Students")
st.write("Upload your resume to find the most suitable job roles and get tips to improve your skills.")

uploaded_files = st.file_uploader("Upload Resume(s) (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Evaluate"):
    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            resume_text = read_file(uploaded_file)
            if resume_text.strip() == "":
                st.warning(f"No text found in {uploaded_file.name}. Skipping...")
                continue

            candidate_name = extract_candidate_name(resume_text, uploaded_file.name)
            top_roles = suggest_job_role(resume_text)

            best_role, best_score = top_roles[0][0], top_roles[0][1]

            # ✅ Use direct skill check for perfect match
            if is_perfect_match(resume_text, best_role):
                best_tips = "✅ Your skills perfectly match this role. No improvement needed!"
            else:
                best_tips = role_tips.get(best_role, "No tips available.")

            top_roles_str = ", ".join([f"{role} ({score:.2f}%)" for role, score in top_roles])
            results.append([candidate_name, top_roles_str, best_role, best_tips])

        results_df = pd.DataFrame(results, columns=["Candidate Name", "Top Role Matches", "Best Suggested Role", "Tips to Improve Skills"])
        st.subheader("📊 Career Guidance Results")
        st.table(results_df)
    else:
        st.warning("Please upload at least one resume.")
