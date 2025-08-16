from fastapi import FastAPI, UploadFile
from transformers import pipeline
import PyPDF2

app = FastAPI()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

CHUNK_SIZE = 100000

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/summarize/")
async def summarize_pdf(file: UploadFile):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file.file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    summaries = []
    for chunk in chunk_text(text):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = " ".join(summaries)
    return {"summary": final_summary}
