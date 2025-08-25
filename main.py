import streamlit as st
from dotenv import load_dotenv
load_dotenv()


pages = {
    "🏠 导航页": [
        st.Page("navigation/Home.py", title="导航页", icon="🚀", default=True),
    ],
    "💬 AI交互平台": [
        st.Page("navigation/Chat.py", title="AI对话平台", icon="💬"),
        st.Page("navigation/ImageGenerator.py", title="文本生成图像", icon="🎨"),
        st.Page("navigation/VideoGenerator.py", title="文本生成视频", icon="📹"),
        st.Page("navigation/Yi_Tradition.py", title="彝脉相承大模型", icon="🏺"),
        st.Page("navigation/RAG.py", title="个人知识库(RAG)", icon="📚"),
    ],
    "🛠️ 工具集合": [
        st.Page("navigation/LightRAG.py", title="知识图谱检索", icon="🕸️"),
        st.Page("navigation/PDFTranslator.py", title="PDF固版翻译", icon="📄"),
        st.Page("navigation/SAM_Segmentor.py", title="分割万物", icon="✂️"),
        st.Page("navigation/PaperPolishing.py", title="论文分段润色", icon="📝"),
        st.Page("navigation/SkySentry.py", title="天眸预警", icon="👁️"),
    ],
    "🤖 Dify应用": [
        st.Page("navigation/Dify.py", title="小红书文案生成", icon="📝"),
    ]
}

pg = st.navigation(pages, expanded=True)
pg.run()

