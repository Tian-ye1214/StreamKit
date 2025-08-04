# -*- coding: utf-8 -*-
import streamlit as st
from pages.Functions.BackendInteraction import BackendInteractionLogic
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING, VISIONMODAL_MAPPING
import asyncio

st.set_page_config(
    page_title="Chat With AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)


async def main():
    backend = BackendInteractionLogic()
    backend.initialize_session_state()
    st.markdown("""
    <h1 style='text-align: center;'>
        Chat With AI
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        await asyncio.gather(backend.user_interaction(), backend.start_new_conversation()
                             , backend.parameter_configuration())
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
            model_display = st.selectbox("选择模型", list(VISIONMODAL_MAPPING.keys()), index=0, help="选择模型")
            st.session_state.model = VISIONMODAL_MAPPING[model_display]

        st.markdown("联系作者")
        st.markdown(f"""
        📧 [Z1092228927@outlook.com](mailto:Z1092228927@outlook.com)<br>
        🐱 [Tian-ye1214](https://github.com/Tian-ye1214)
        """, unsafe_allow_html=True)

    if sections == '视觉对话':
        backend.image_upload()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("在这里输入您的问题："):
        await asyncio.gather(backend.user_input(prompt), backend.search_interaction())

        with st.chat_message("assistant"):
            try:
                with st.spinner('模型思考中'):
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
