# main.py
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI(title="Live Autocorrect API")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


model_path = "fine_tuned_t5"  # Adjust to your model folder location
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_sentences(text: str):
    """
    Splits text into sentences using ., ?, or ! as delimiters, preserving punctuation.
    """
    sentences = re.findall(r'[^.?!]+[.?!]+', text)
    remainder = re.sub(r'[^.?!]+[.?!]+', '', text).strip()
    if remainder:
        sentences.append(remainder)
    return [s.strip() for s in sentences if s.strip()]

def infer_live_autocorrect(text: str, max_length: int = 128) -> dict:
    """
    Extracts up to the last three sentences (with punctuation) from the input text,
    prepends "autocorrect: ", and runs the model.
    Returns a dictionary with:
      - corrected_block: full corrected text block.
      - corrected_last_sentence: corrected version of the last sentence.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {"corrected_block": "", "corrected_last_sentence": ""}
    selected_sentences = sentences[-3:]
    prompt_input = "autocorrect: " + " ".join(selected_sentences)
    inputs = tokenizer.encode(prompt_input, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected_sentences = split_sentences(corrected_text)
    corrected_last_sentence = corrected_sentences[-1] if corrected_sentences else corrected_text
    return {"corrected_block": corrected_text, "corrected_last_sentence": corrected_last_sentence}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        result = infer_live_autocorrect(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})