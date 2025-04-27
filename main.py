import streamlit as st
from dotenv import load_dotenv
load_dotenv()

pages = {
    "AI交互平台": [
        st.Page("navigation/Chat.py", title="AI对话平台"),
        st.Page("navigation/ImageGenerator.py", title="文本生成图像"),
    ],
    "Dify": [
        st.Page("navigation/Dify.py", title="小红书文案生成"),
    ],
    "Tools": [
        st.Page("navigation/LightRAG.py", title="知识图谱检索"),
        st.Page("navigation/PDFTranslator.py", title="PDF固版翻译"),
        st.Page("navigation/SAM_Segmentor.py", title="分割万物"),
        st.Page("navigation/PaperPolishing.py", title="论文分段润色"),
        st.Page("navigation/SkySentry.py", title="天眸预警"),
        st.Page("navigation/RAG.py", title="知识库"),
    ],
}

pg = st.navigation(pages)
pg.run()
