import os
from typing import Dict, Tuple, Optional
import requests

# --------- palavras ignoradas durante processo NLP ---------
PT_STOPWORDS = {
    "a","o","as","os","de","da","do","das","dos","para","por","em","no","na","nos","nas","um","uma",
    "e","ou","mas","que","com","à","ao","aos","às","se","sua","seu","suas","seus","eu","você","vocês",
    "já","não","sim","como","sobre","mais","menos","meu","minha","meus","minhas","obrigado","obrigada"
}

# --------- Classificação Heuristic detecta e-mail é produtivo ou inprodutivo ---------
PRODUCTIVE_HINTS = {
    "status","andamento","progresso","atualização","atualizacao","prazo","chamado","ticket","suporte",
    "erro","bug","falha","acesso","senha","login","fatura","boleto","contrato","api","requisição",
    "requisicao","solicito","solicitação","solicitacao","anexo","arquivo","documento","evidência","evidencia"
}
UNPRODUCTIVE_HINTS = {
    "feliz","parabéns","parabens","natal","ano novo","bom dia","boa tarde","boa noite","agradeço","agradeco",
    "obrigado","obrigada","atenciosamente","abraços","abrazos","obg"
}
# -------------------------------------------

def filtra_stopwords(text: str) -> str:
    words = text.lower().split()
    filtered_words = [word for word in words if word not in PT_STOPWORDS]
    return ' '.join(filtered_words)

def Classificacao_heuristica(text: str) -> Tuple[str, float, str]:

    filtered_text = filtra_stopwords(text)
    t = filtered_text.lower()
    
    p_score = sum(int(h in t) for h in PRODUCTIVE_HINTS)
    u_score = sum(int(h in t) for h in UNPRODUCTIVE_HINTS)
    
    if p_score == u_score:
        p_score += 1 if len(filtered_text) > 100 else 0
    
    if p_score >= u_score:
        conf = min(0.95, 0.55 + 0.05 * p_score)
        return "Produtivo", conf, "heuristic"
    else:
        conf = min(0.95, 0.55 + 0.05 * u_score)
        return "Improdutivo", conf, "heuristic"

# --------- Classificação via Hugging Face ---------
def hf_zero_shot(text: str) -> Optional[Tuple[str,float,str]]:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        return None
    try:
        filtered_text = filtra_stopwords(text)
        
        # Utiliza modelo multilingue
        model = "joeddav/xlm-roberta-large-xnli"
        url = f"https://api-inference.huggingface.co/models/{model}"
        payload = {
            "inputs": filtered_text,
            "parameters": {"candidate_labels": ["Produtivo", "Improdutivo"]},
        }
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        labels = data.get("labels", [])
        scores = data.get("scores", [])
        if labels and scores:
            best = max(zip(labels, scores), key=lambda x: x[1])
            return best[0], float(best[1]), "hf-zero-shot"
    except Exception:
        return None
    return None

# --------- Resposta LLM  ---------
def Resposta_LLM(category: str, text: str) -> str:
    lower = text.lower()
    
    if category == "Produtivo":
        if any(k in lower for k in ["status", "andamento", "ticket", "chamado"]):
            return ("Olá! Recebemos sua solicitação de status e estamos verificando o andamento. "
                    "Em breve retornaremos com a atualização completa do ticket. "
                    "Se houver algum prazo crítico, por favor, responda este e‑mail informando.")
        if any(k in lower for k in ["anexo", "documento", "arquivo"]):
            return ("Olá! O documento foi recebido com sucesso. Vamos validar o conteúdo e responder "
                    "com os próximos passos. Caso falte algo, entraremos em contato.")
        if any(k in lower for k in ["erro","acesso","senha","login","falha","bug"]):
            return ("Olá! Identificamos seu relato de erro. Abrimos/atualizamos o chamado junto à equipe técnica "
                    "e enviaremos novos informes assim que houver correção ou workaround disponível.")
        return ("Olá! Sua solicitação foi registrada e está em análise. "
                "Retornaremos com os desdobramentos ou próximos passos em breve.")

    # Improdutivo
    if any(k in lower for k in ["feliz","parabéns","parabens","natal","ano novo"]):
        return ("Muito obrigado pela mensagem e pelos votos! "
                "Agradecemos o contato e permanecemos à disposição.")
    return ("Obrigado pela mensagem! Caso necessite de suporte ou tenha alguma solicitação, "
            "por favor responda este e‑mail com os detalhes para que possamos ajudar.")

def Classifica_responde(text: str) -> Dict:

    for fn in (hf_zero_shot,):
        result = fn(text)
        if result:
            label, conf, provider = result
            return {
                "category": label,
                "confidence": round(conf, 3),
                "provider": provider,
                "response": Resposta_LLM(label, text)
            }

    label, conf, provider = Classificacao_heuristica(text)
    return {
        "category": label,
        "confidence": round(float(conf), 3),
        "provider": provider,
        "response": Resposta_LLM(label, text)
    }