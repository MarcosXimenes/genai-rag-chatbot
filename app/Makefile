# ==============================================================================
# ⚙️ Configurações do Ambiente Virtual
# ==============================================================================
-include .env
export

VENV := .venv
ACTIVATE := $(VENV)/bin/activate
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

.DEFAULT_GOAL := install

.PHONY: install run-dev clear code-formatting setup-env

# ==============================================================================
# 🚀 Comandos Principais
# ==============================================================================

## install: Cria o ambiente virtual e instala as dependências.
install: $(ACTIVATE)

$(ACTIVATE): requirements.txt
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	. $(ACTIVATE); $(PIP) install --upgrade pip
	. $(ACTIVATE); $(PIP) install -r requirements.txt
	@touch $(ACTIVATE)
	@echo "✅ Virtual environment ready and dependencies installed."

## setup-env: Cria um arquivo .env.example como template.
setup-env:
	@echo "✨ Creating .env.example file..."
	@echo '## ⚙️ General Project Settings' > .env.example
	@echo 'PROJECT_ID=your-gcp-project-id' >> .env.example
	@echo 'SERVICE=genai-rag-assistant' >> .env.example
	@echo '' >> .env.example
	@echo '## ☁️ Google Cloud & Vertex AI Settings' >> .env.example
	@echo 'GOOGLE_CLOUD_PROJECT=your-gcp-project-id' >> .env.example
	@echo 'GOOGLE_CLOUD_LOCATION=us-central1' >> .env.example
	@echo 'GOOGLE_GENAI_USE_VERTEXAI=True' >> .env.example
	@echo 'GOOGLE_APPLICATION_CREDENTIALS=./credentials/credential.json' >> .env.example
	@echo '' >> .env.example
	@echo '## 🤖 AI Model Settings' >> .env.example
	@echo 'GEMINI_EMBEDDING_MODEL=gemini-embedding-001' >> .env.example
	@echo 'GEMINI_EMBEDDING_DIMENSIONALITY=2048' >> .env.example
	@echo 'GEMINI_QA_MODEL=gemini-2.5-pro' >> .env.example
	@echo 'OPENAI_API_KEY=sk-your-openai-api-key' >> .env.example
	@echo 'OPENAI_QA_MODEL=gpt-4o' >> .env.example
	@echo "📄 Done. Please copy to .env and fill in your values."

## run-dev: Inicia o servidor de desenvolvimento com as variáveis do .env.
run-dev: install
	@# Verifica se o arquivo .env existe antes de tentar iniciar.
	@[ -f .env ] || { \
		echo "⚠️  Error: .env file not found."; \
		echo "Please run 'make setup-env', then copy '.env.example' to '.env' and fill it out."; \
		exit 1; \
	}
	@echo "🚀 Starting development server..."
	@# Carrega as variáveis do .env e executa o Uvicorn
	set -a; source .env; set +a; \
	uvicorn main:app --timeout-keep-alive 60 --port 8080 --reload

## deploy-cloud-run: Faz o deploy da aplicação no Google Cloud Run.
deploy-cloud-run:
	gcloud run deploy $(SERVICE) \
		--source . \
		--region $(GOOGLE_CLOUD_LOCATION) \
		--allow-unauthenticated \
		--project=$(GOOGLE_CLOUD_PROJECT)
	@echo "✅ Deployment complete."

## code-formatting: Formata o código com isort, black e ruff.
code-formatting: install
	@echo "💅 Formatting code..."
	. $(ACTIVATE); isort .
	. $(ACTIVATE); black .
	@echo "✅ Code formatting complete."

## clear: Remove arquivos de cache e temporários do Python.
clear:
	@echo "🧹 Cleaning up project..."
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find . -type d -name '.ruff_cache' -exec rm -rf {} +
	@echo "🗑️ Cache files removed."