import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import fitz

sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

jd_paths = ["jd_web_developer.txt", "jd_web_developer.txt", "jd_web_developer.txt", "jd_web_developer.txt", "Job Description.txt"]
resume_paths = ["Hardik_Hans_resume_3.pdf", "Hardik_Hans_resume_4.pdf", "Hardik_Hans_resume_5.pdf", "dipankar_resume.pdf","aryancv1.pdf"]

jds_embeddings = [sentence_transformer_model.encode(extract_text_from_pdf(path)) for path in jd_paths]
resumes_embeddings = [sentence_transformer_model.encode(extract_text_from_pdf(path)) for path in resume_paths]

jds_embeddings_np = np.array(jds_embeddings)
resumes_embeddings_np = np.array(resumes_embeddings)

print("Shape of jds_embeddings:", jds_embeddings_np.shape)
print("Shape of resumes_embeddings:", resumes_embeddings_np.shape)

jds_train, jds_val, resumes_train, resumes_val = train_test_split(jds_embeddings_np, resumes_embeddings_np, test_size=0.4, random_state=42)

jd_embeddings_train = jds_train
resume_embeddings_train = resumes_train

print("Shape of jd_embeddings_train:", jd_embeddings_train.shape)
print("Shape of resume_embeddings_train:", resume_embeddings_train.shape)

a, b = jd_embeddings_train.shape
print(a, b)

jd_embeddings_train = np.expand_dims(jd_embeddings_train, axis=1)

model = models.Sequential([
    layers.LSTM(resume_embeddings_train.shape[1], input_shape=(1, b)),
    layers.Dense(resume_embeddings_train.shape[1], activation='relu'),
    layers.Dense(resume_embeddings_train.shape[1], activation='sigmoid')  
])

model.compile(optimizer='adam', loss='mse') 

model.fit(x=jd_embeddings_train, y=resume_embeddings_train, epochs=10, batch_size=32)

def rank_resumes(job_description, resumes):
    for jd_embedding in job_description:
        jd_scores = []
        for resume_embedding in resumes:
            similarity = np.dot(jd_embedding, resume_embedding) / (np.linalg.norm(jd_embedding) * np.linalg.norm(resume_embedding))
            jd_scores.append(similarity)
        print(jd_scores)
    jd_embedding = job_description
    resume_embeddings = resumes
    similarities = cosine_similarity(jd_embedding, resume_embeddings)
    ranked_resumes_indexes = np.argsort(similarities)[::-1]
    return ranked_resumes_indexes, similarities

ranked_resumes_indexes, similarities = rank_resumes(jds_val, resumes_val)

print("Ranked Resumes:")
print(similarities)