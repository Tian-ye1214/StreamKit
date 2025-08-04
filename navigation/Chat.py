# -*- coding: utf-8 -*-
import streamlit as st
from pages.Functions.BackendInteraction import BackendInteractionLogic
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING, VISIONMODAL_MAPPING
import asyncio
import random

st.set_page_config(
    page_title="Chat With AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

Greeting_Template = [
    "今天需要我协助处理什么事项吗？",
    "有什么我可以跟进的吗？",
    "有什么需要我准备的吗？",
    "有新的进展或问题需要讨论吗？",
    "我这边随时待命，有需要请说。",
    "有什么我能帮上忙的吗？",
    "今天一切顺利吗？",
    "今天有什么议程？",
    "您在忙什么？",
    "您今天在想什么？",
]


async def main():
    Assistant_placeholder = st.empty()
    backend = BackendInteractionLogic()
    backend.initialize_session_state()

    with st.sidebar:
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

        await asyncio.gather(backend.user_interaction(), backend.start_new_conversation()
                             , backend.parameter_configuration())

        st.markdown("联系作者")
        st.markdown(f"""
        📧 [Z1092228927@outlook.com](mailto:Z1092228927@outlook.com)<br>
        🐱 [Tian-ye1214](https://github.com/Tian-ye1214)
        """, unsafe_allow_html=True)

    if sections == '视觉对话':
        backend.image_upload()

    for message in st.session_state.chat_messages:
        avatar = "😀" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("在这里输入您的问题："):
        await asyncio.gather(backend.user_input(prompt), backend.search_interaction())

        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner('模型思考中'):
                    await asyncio.gather(backend.ai_generation(sections))
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")

    flag = st.session_state.get('messages', None)
    if flag is not None and flag != []:
        Assistant_placeholder.markdown(f"""
        <h4 style='text-align: left; margin-top: 10px; margin-bottom: 10px; font-size: 16px;'>
            当前Assistant:{model_display}
        </h4>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
                    <style>
                    .stChatMessage[data-testid="stChatMessage"] {
                        background-color: #fdfdf8 !important;
                        border-radius: 8px;
                        margin: 5px 0;
                        padding: 10px;
                    }

                    .stChatMessage[data-testid="stChatMessage"] p {
                        font-size: 24px !important;
                    }

                    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
                            flex-direction: row-reverse;
                            text-align: right;
                        }

                    .stBottom {
                        position: fixed;
                        top: 50%;
                        left: 60%;
                        bottom: auto;
                        transform: translate(-50%, -50%);
                        width: 100%;
                        max-width: 1000px;
                        background: #fdfdf8;
                    }
                    </style>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
        <h1 style='text-align: center;'>
            {random.choice(Greeting_Template)}
        </h1>
        <div style='text-align: center; margin-bottom: 20px;'>
        </div>
        """, unsafe_allow_html=True)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'chat'
current_page = 'chat'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page

asyncio.run(main())
