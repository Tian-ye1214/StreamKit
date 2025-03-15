import streamlit as st
from dotenv import load_dotenv
load_dotenv()

pages = {
    "AI交互平台": [
        st.Page("navigation/Chat.py", title="AI对话平台"),
        st.Page("navigation/MultimodalChat.py", title="多模态AI交互平台"),
    ],
    "Dify": [
        st.Page("navigation/Dify.py", title="小红书文案生成"),
    ],
    "Tools": [
        st.Page("navigation/LightRAG.py", title="知识图谱检索"),
        st.Page("navigation/PDFTranslator.py", title="PDF翻译"),
        st.Page("navigation/SAM_Segmentor.py", title="分割万物"),
    ],
}

pg = st.navigation(pages)
pg.run()
