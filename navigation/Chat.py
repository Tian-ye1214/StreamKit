# -*- coding: utf-8 -*-
import streamlit as st
from pages.Functions.BackendInteraction import BackendInteractionLogic
from pages.Functions.Constants import (
    HIGHSPEED_MODEL_MAPPING,
    REASON_MODELS,
    VISIONMODAL_MAPPING
)
import asyncio

st.set_page_config(
    page_title="Chat With AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


async def main():
    backend = BackendInteractionLogic()
    await backend.initialize_session_state()
    st.markdown("""
    <h1 style='text-align: center;'>
        Chat With AI
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **欢迎来到未来对话体验** 🌟
        
        💡 **快速上手**
        1. 在侧边栏选择心仪模型
        2. 对话框直接输入问题
        3. 见证AI的创意迸发
        
        🎨 **个性化设置**\n
        ✅ 多个模型自由切换<br>
        ✅ 温度参数实时调节<br>
        ✅ 上下文记忆深度定制<br>

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 20px;">
            🎭 试试让AI：<br>
            • 用莎士比亚风格写周报<br>
            • 用幼儿园术语解释量子纠缠<br>
            • 为你的创业计划提供风险评估<br>
            每一次对话都是新的冒险！
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        await asyncio.gather(backend.user_interaction(), backend.start_new_conversation(), backend.parameter_configuration())

        st.markdown("""
        <h3 style='text-align: center;'>
            模型配置
        </h3>
        """, unsafe_allow_html=True)
        sections = st.radio("对话模式",
                            ["文本对话", "视觉对话"], 
                            index=0,
                            )
        if sections == '文本对话':
            model_display = st.selectbox("选择模型", list(HIGHSPEED_MODEL_MAPPING.keys()), index=0, help="选择模型")
            st.session_state.model = HIGHSPEED_MODEL_MAPPING[model_display]
        else:
            model_display = st.selectbox("选择模型", list(VISIONMODAL_MAPPING.keys()), index=1, help="选择模型")
            st.session_state.model = VISIONMODAL_MAPPING[model_display]

        if st.session_state.model not in REASON_MODELS:
            st.session_state.system_prompt = "You are a helpful assistant."

        st.markdown("联系作者")
        st.markdown(f"""
        📧 [Z1092228927@outlook.com](mailto:Z1092228927@outlook.com)<br>
        🐱 [Tian-ye1214](https://github.com/Tian-ye1214)
        """, unsafe_allow_html=True)

    if sections == '视觉对话':
        await backend.image_upload()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("在这里输入您的问题："):
        await asyncio.gather(backend.user_input(prompt), backend.search_interaction())

        with st.chat_message("assistant"):
            try:
                await asyncio.gather(backend.ai_generation(sections))
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'chat'
current_page = 'chat'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page

asyncio.run(main())
