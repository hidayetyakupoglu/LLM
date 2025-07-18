# Multi-Document Q&A with Free LLM (OpenRouter + Mistral)

Bu proje, PDF/TXT/DOCX dosyalarınızı yükleyip içlerinden doğal dilde yanıtlar alabileceğiniz bir **Streamlit uygulamasıdır**.

## ⚙️ Kurulum

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 OpenRouter API

`.streamlit/secrets.toml` içeriği:
```
[general]
openrouter_key = "YOUR_OPENROUTER_API_KEY"
```

## 🚀 Çalıştırma
```bash
streamlit run app.py
```
