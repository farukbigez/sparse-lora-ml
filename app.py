# app.py
# Purpose: Streamlit chat interface for the German Teacher AI.

import streamlit as st
import requests

# ===============================================
# 1. PAGE CONFIG
# ===============================================
st.set_page_config(page_title="🇩🇪 German Teacher AI", page_icon="🇩🇪", layout="wide")
st.title("🇩🇪 Yapay Zeka ile Almanca Öğren")
st.caption("Pruning + LoRA ile optimize edilmiş, RAG ile güçlendirilmiş Qwen modeli.")

# ===============================================
# 2. SESSION STATE (Chat History)
# ===============================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hallo! Almanca ile ilgili sorularını sorabilir veya bir cümle yazıp düzeltmemi isteyebilirsin."}
    ]

# ===============================================
# 3. DISPLAY MESSAGES
# ===============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================================
# 4. USER INPUT & API CALL
# ===============================================
if prompt := st.chat_input("Almanca bir şeyler yaz veya soru sor..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor ve kuralları taranıyor..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"query": prompt},
                    timeout=60,
                )
                if response.status_code == 200:
                    reply = response.json()["response"]
                else:
                    reply = f"❌ API Hatası: {response.status_code} - {response.text}"
            except requests.exceptions.ConnectionError:
                reply = "❌ API sunucusuna bağlanılamıyor. `python src/06_api.py` ile sunucuyu başlattığınızdan emin olun."
            except Exception as e:
                reply = f"❌ Beklenmeyen hata: {e}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})