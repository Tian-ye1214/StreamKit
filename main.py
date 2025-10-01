import streamlit as st
from dotenv import load_dotenv
load_dotenv()


pages = {
    "🏠 导航页": [
        st.Page("navigation/Home.py", title="导航页", icon="🚀", default=True),
    ],
    "📜 言语洞澈(文本类应用)": [
        st.Page("navigation/Chat.py", title="AI对话平台", icon="💬"),
        st.Page("navigation/RAG.py", title="个人知识库(RAG)", icon="📚"),
        st.Page("navigation/PDFTranslator.py", title="PDF固版翻译", icon="📄"),
        st.Page("navigation/PaperPolishing.py", title="论文分段润色", icon="📝"),
        st.Page("navigation/Dify.py", title="小红书文案生成", icon="📝"),
    ],
    "🔭 新域探微(研究类应用)": [
        st.Page("navigation/Nuosu.py", title="彝脉相承大模型", icon="🏺"),
        st.Page("navigation/AncientBuilding.py", title="古建筑图像生成", icon="🏯"),
        st.Page("navigation/SkySentry.py", title="天眸预警", icon="👁️"),
        st.Page("navigation/LightRAG.py", title="知识图谱检索", icon="🕸️"),
    ],
    "🌌 融象观言(多模态类应用)": [
        st.Page("navigation/ImageGenerator.py", title="图像生成", icon="🎨"),
        st.Page("navigation/VideoGenerator.py", title="视频生成", icon="📹"),
        st.Page("navigation/SAM_Segmentor.py", title="分割万物", icon="✂️"),
    ],
    "📬 反馈与建议": [
        st.Page("navigation/suggest.py", title="意见与建议", icon="📝"),
    ],
}

pg = st.navigation(pages, expanded=True)
pg.run()

