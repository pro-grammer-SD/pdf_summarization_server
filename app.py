import gradio as gr
from transformers import pipeline
import PyPDF2

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

CHUNK_SIZE = 100000

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    summaries = []
    for chunk in chunk_text(text):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

interface = gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="text",
    title="PDF Summarizer",
    description="Upload a PDF and get a summary using BART Large CNN."
)

interface.launch()
