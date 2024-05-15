import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import fitz

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

jd_paths = ["Job Description.txt"]
resume_paths = ["Hardik_Hans_resume_4.pdf", "Hardik_Hans_resume_5.pdf","dipankar_resume.pdf","aryancv1.pdf"]

jds_embeddings = [model.encode(extract_text_from_pdf(path)) for path in jd_paths]
resumes_embeddings = [model.encode(extract_text_from_pdf(path)) for path in resume_paths]

similarity_scores = []
for jd_embedding in jds_embeddings:
    jd_scores = []
    for resume_embedding in resumes_embeddings:
        similarity = np.dot(jd_embedding, resume_embedding) / (np.linalg.norm(jd_embedding) * np.linalg.norm(resume_embedding))
        jd_scores.append(similarity)
    similarity_scores.append(jd_scores)

for i, jd_path in enumerate(jd_paths):
    print(f"JD: {jd_path}")
    for j, resume_path in enumerate(resume_paths):
        print(f"Resume: {resume_path}, Similarity: {similarity_scores[i][j]}")
    print()
