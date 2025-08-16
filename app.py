import gradio as gr
import PyPDF2
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

CHUNK_SIZE = 50000

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    summaries = []
    for chunk in chunk_text(text):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="text",
    title="PDF Summarizer",
    description="Upload a PDF and get a summary using a small local BART model."
).launch(server_name="0.0.0.0", server_port=7860)
