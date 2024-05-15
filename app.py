import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

resume_paths = [
    "Hardik_Hans_resume_3.pdf",
    "Hardik_Hans_resume_4.pdf",
    "Hardik_Hans_resume_5.pdf"
]

query_path = "Job Description.txt"

resumes = [extract_text_from_pdf(path) for path in resume_paths]
query = extract_text_from_pdf(query_path)

resume_embeddings = model.encode(resumes)
query_embedding = model.encode(query)

similarities = cosine_similarity([query_embedding], resume_embeddings)[0]

sorted_indexes = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

print("Ranked Resumes:")
for i in sorted_indexes:
    print(f"Similarity: {similarities[i]:.4f}\nResume: {resume_paths[i]}\n")
