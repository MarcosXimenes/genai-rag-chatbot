import os

from dotenv import load_dotenv

load_dotenv()

## ⚙️ Configurações Gerais do Projeto
# -------------------------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "ximenes-sandbox")
SERVICE = os.getenv("SERVICE", "genai-rag-assistant")


## ☁️ Configurações do Google Cloud e Vertex AI
# -------------------------------------------------------------------------------
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = (
    os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"
)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", "./credentials/credential.json"
)

## 🤖 Configurações dos Modelos de IA (Gemini e OpenAI)
# -------------------------------------------------------------------------------

# --- Modelos Gemini (Google) ---
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIMENSIONALITY = int(
    os.getenv("GEMINI_EMBEDDING_DIMENSIONALITY", "768")
)
GEMINI_QA_MODEL = os.getenv("GEMINI_QA_MODEL", "gemini-1.5-pro-latest")

# --- Modelos OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    print("⚠️  Aviso: A variável de ambiente OPENAI_API_KEY não está definida.")
