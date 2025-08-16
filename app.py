import gradio as gr
import PyPDF2
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn
import tempfile

# --- Summarization setup ---
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name, device=-1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_TOKENS = 1024

def chunk_text_by_tokens(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def summarize_pdf(pdf_file_path):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    summaries = []
    for chunk in chunk_text_by_tokens(text):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

# --- Gradio Interface ---
def gradio_summarizer(pdf_file):
    return summarize_pdf(pdf_file.name)

gr_interface = gr.Interface(
    fn=gradio_summarizer,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="text",
    title="PDF Summarizer",
    description="Upload a PDF and get a summary using DistilBART."
)

# --- FastAPI Endpoint ---
app = FastAPI()

@app.post("/summarize/")
async def summarize_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        summary = summarize_pdf(tmp.name)
    return PlainTextResponse(summary)

# --- Launch Gradio + FastAPI ---
if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: gr_interface.launch(server_name="0.0.0.0", server_port=7860, share=True)).start()
    uvicorn.run(app, host="0.0.0.0", port=8500)
