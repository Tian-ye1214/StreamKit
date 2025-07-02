import streamlit as st

# 页面配置
st.set_page_config(
    page_title="StreamKit -- AI Nexus",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# 自定义CSS样式
def load_css():
    st.markdown("""
    <style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #0f1419, #1a1f2e, #0f1419);
        color: #e1e5e9;
    }
    
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(138, 180, 248, 0.3);
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        text-align: center;
        font-weight: 300;
        letter-spacing: 1px;
        margin-bottom: 3rem;
    }
    
    /* 卡片样式 */
    .nav-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        position: relative;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        min-height: 280px;
        display: flex;
        flex-direction: column;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        border-color: rgba(138, 180, 248, 0.5);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 
                    0 0 20px rgba(138, 180, 248, 0.1);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
        text-align: center;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    
    .card-subtitle {
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 1rem;
        font-style: italic;
    }
    
    .card-description {
        font-size: 0.85rem;
        line-height: 1.6;
        color: #c9d1d9;
        flex-grow: 1;
        margin-bottom: 1rem;
    }
    
    /* 特定卡片主题 */
    .platform { border-left: 4px solid #58a6ff; }
    .platform .card-icon { color: #58a6ff; }
    
    .dify-section { border-left: 4px solid #bc8cff; }
    .dify-section .card-icon { color: #bc8cff; }
    
    .tools-section { border-left: 4px solid #ff7b72; }
    .tools-section .card-icon { color: #ff7b72; }
    
    /* 进度条 */
    .progress-bar {
        width: 100%;
        height: 6px;
        background: rgba(48, 54, 61, 0.5);
        border-radius: 3px;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .progress {
        height: 100%;
        background: linear-gradient(90deg, #a5a3ff, #58a6ff);
        border-radius: 3px;
        width: 30%;
        animation: progressFill 2s ease-in-out;
    }
    
    @keyframes progressFill {
        from { width: 0%; }
        to { width: 30%; }
    }
    
    /* 结构图标 */
    .structure-icons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        opacity: 0.6;
        justify-content: center;
    }
    
    .structure-icons i {
        font-size: 1.2rem;
        color: #56d4dd;
    }
    
    /* Streamlit按钮自定义 */
    .stButton > button {
        background: rgba(48, 54, 61, 0.7);
        border: 1px solid rgba(138, 180, 248, 0.3);
        color: #c9d1d9;
        border-radius: 6px;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.2rem 0;
    }
    
    .stButton > button:hover {
        background: rgba(138, 180, 248, 0.1);
        border-color: rgba(138, 180, 248, 0.5);
        color: #ffffff;
        transform: translateX(5px);
    }
    
    /* 动画效果 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .nav-card {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    /* 隐藏Streamlit的默认边距 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    /* 粒子效果容器 */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: rgba(138, 180, 248, 0.3);
        border-radius: 50%;
        animation: particleFloat 15s linear infinite;
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
        }
    }
    </style>
    
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)


# 背景粒子效果
def add_particles():
    particles_html = """
    <div class="particles" id="particles"></div>
    <script>
    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        
        const particlesContainer = document.getElementById('particles');
        if (particlesContainer) {
            particlesContainer.appendChild(particle);
            
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.parentNode.removeChild(particle);
                }
            }, 20000);
        }
    </script>
    """
    st.markdown(particles_html, unsafe_allow_html=True)


# 创建卡片组件
def create_card(card_class, icon, title, subtitle, description, buttons=None, extra_content=None):
    card_html = f"""
    <div class="nav-card {card_class}">
        <div class="card-icon">
            <i class="{icon}"></i>
        </div>
        <h3 class="card-title">{title}</h3>
        <p class="card-subtitle">{subtitle}</p>
        <div class="card-description">
            {description}
        </div>
        {extra_content if extra_content else ''}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    if buttons:
        for button_text, page_name in buttons:
            if st.button(button_text, key=f"btn_{page_name}_{card_class}"):
                st.switch_page(f"navigation/{page_name}")


# 主页面内容
def main():
    load_css()
    add_particles()

    st.markdown('<h1 class="main-title">StreamKit -- AI Nexus</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Witness the future of AI</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # 第一行
    with col1:
        create_card(
            "platform",
            "fas fa-terminal",
            "AI交互平台",
            "Current: AI Platform",
            "感受AI能力边界<br>Explore the Boundaries of AI Abilities",
            [
                ("AI对话平台", "Chat.py"),
                ("文本生成图像", "ImageGenerator.py"),
                ("彝脉相承大模型", "Yi_Tradition.py"),
                ("个人知识库(RAG)", "RAG.py"),
            ]
        )

    with col2:
        create_card(
            "tools-section",
            "fas fa-tools",
            "工具集合",
            "Tool Collection",
            "专业工具和实用功能集合<br>Professional tools and utility functions",
            [
                ("知识图谱检索", "LightRAG.py"),
                ("PDF固版翻译", "PDFTranslator.py"),
                ("分割万物", "SAM_Segmentor.py"),
                ("论文分段润色", "PaperPolishing.py"),
                ("天眸预警", "SkySentry.py"),
            ]
        )

    with col3:
        create_card(
            "dify-section",
            "fas fa-robot",
            "Dify 应用",
            "Dify Applications",
            "基于 Dify 平台的智能应用<br>Smart applications based on Dify platform",
            [
                ("小红书文案生成", "Dify.py")
            ]
        )
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 底部信息
    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 0.8rem; margin-top: 2rem;">
        <p>🚀 享受未来感的AI界面导航体验！</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
