# app.py
# Streamlit chat interface for the German Teacher AI.

import streamlit as st
import requests

st.set_page_config(page_title="🇩🇪 German Teacher AI (7B)", page_icon="🇩🇪", layout="wide")
st.title("🇩🇪 Yapay Zeka ile Almanca Öğren (7B + RAG)")
st.caption("Mevcut model: Qwen2.5-7B-Instruct | Pruning + LoRA + RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hallo! Almanca ile ilgili sorularını sorabilir veya bir cümle yazıp düzeltmemi isteyebilirsin."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Almanca bir şeyler yaz veya soru sor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor ve kuralları taranıyor..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",   # Change if API runs elsewhere
                    json={"query": prompt},
                    timeout=120,
                )
                if response.status_code == 200:
                    reply = response.json()["response"]
                else:
                    reply = f"❌ API Hatası: {response.status_code}"
            except Exception as e:
                reply = f"❌ Bağlantı hatası: {e}"
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})