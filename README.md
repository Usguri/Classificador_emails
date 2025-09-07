# Classificador de e-mails (PT-BR)

Aplicação web simples para **classificar emails (Produtivo/Improdutivo)** e **sugerir respostas automáticas**.
Funciona **100% offline** (baseline heurístico) e permite **upgrade opcional** para IA via
**Hugging Face Inference API** (zero-shot).

## Demo local

### Requisitos
- Python 3.11+
- (Opcional) Token do Hugging Face para melhorar a classificação (`HUGGINGFACE_API_TOKEN`).

### Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Acesse: http://localhost:8000

### Variáveis de ambiente (opcionais)
- `HUGGINGFACE_API_TOKEN`: Habilita zero-shot (`joeddav/xlm-roberta-large-xnli`) para melhor qualidade.

## Como funciona

- **Pré-processamento:** normalização de texto, remoção de ruído, stopwords básicas em PT-BR.
- **Classificação (ordem de tentativa):**
  1. `hf_zero_shot` (se `HUGGINGFACE_API_TOKEN` estiver definido).
  2. `heuristic_classify`: regras por palavras‑chave para decisão de fallback.
- **Geração de resposta:** templates por categoria (produtivo vs improdutivo) com variações por intenção (status, anexo, erro).

## Estrutura

```
.
├── app
│   ├── main.py                # FastAPI + rotas
│   ├── nlp.py                 # pipeline de NLP e classificação
│   ├── templates
│   │   └── index.html         # interface
│   └── static
│       └── styles.css         # estilo
├── requirements.txt
└── README.md
```

## Deploy sugerido (gratuito)
- **Render** ou **Hugging Face Spaces (Gradio/Streamlit/FastAPI com `uvicorn`)**: subir repositório, definir build `pip install -r requirements.txt` e start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **Railway/Replit**: similar; configure variável `PORT` se necessário.
- **Vercel**: não recomendado para FastAPI puro; considere adapter/serverless frameworks ou migre o frontend para Vercel e mantenha o backend em Render/Railway.

## Licença
MIT