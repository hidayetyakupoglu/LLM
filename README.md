# Multi-Document Q&A with Free LLM (OpenRouter + Mistral)

Bu proje, PDF/TXT/DOCX dosyalarınızı yükleyip içlerinden doğal dilde yanıtlar alabileceğiniz bir **Streamlit uygulamasıdır**.

## ⚙️ Kurulum

```bash
git clone https://github.com/USERNAME/multi-doc-llm-streamlit.git
cd multi-doc-llm-streamlit
python -m venv venv
source venv/bin/activate  # veya Windows için venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 OpenRouter API

Streamlit secrets dosyasına API anahtarınızı girin:

`.streamlit/secrets.toml`:
```
[general]
openrouter_key = "YOUR_OPENROUTER_API_KEY"
```

## 🚀 Çalıştırma
```bash
streamlit run app.py
```
