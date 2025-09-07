import io
import os
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from pdfminer.high_level import extract_text
from app.nlp import Classifica_responde

app = FastAPI(title="Classificador de emails", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify", response_class=JSONResponse)
async def classify_endpoint(text: str = Form(None), file: UploadFile = None):
    extraiconteudo = ""
    if file is not None:
        extraiconteudo = read_file_content(file)
    text = (text or "").strip()
    payload = text or extraiconteudo
    if not payload:
        return JSONResponse({"error": "Nenhum texto foi enviado."}, status_code=400)    
    result = Classifica_responde(payload)
    result["chars"] = len(payload)

    return JSONResponse(result)

def read_file_content(file: UploadFile) -> str:
    filename = file.filename or ""
    content = file.file.read()
    if filename.lower().endswith(".pdf"):
        return extract_text(io.BytesIO(content)) or ""
    
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""