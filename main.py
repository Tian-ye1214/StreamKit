# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI
from pages.Functions.ExtractFileContents import encode_image_to_base64
from pages.Functions.BackendInteraction import (
    UserInteraction,
    ParameterConfiguration,
    get_system_prompt,
    initialize_session_state)
from pages.Functions.WebSearch import WebSearch
from pages.Functions.Constants import (
    MODEL_MAPPING,
    VISIONMODAL_MAPPING,
    SEARCH_METHODS,
    REASON_MODELS,
)

st.set_page_config(
    page_title="Chat With AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    initialize_session_state()
    st.session_state.openai_client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)
    uploaded_image = None
    st.markdown("""
    <h1 style='text-align: center;'>
        Chat With AI
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        UserInteraction()

        st.markdown("""
        <h3 style='text-align: center;'>
            模型配置
        </h3>
        """, unsafe_allow_html=True)

        # interaction_mode = st.radio(
        #     "交互模式",
        #     ["纯文本模式", "多模态模式"],
        #     index=0,
        #     help="选择交互模式：纯文本或多模态（支持图片+文本）"
        # )
        interaction_mode = '纯文本模式'

        if st.button("开启新对话", help="开启新对话将清空当前对话记录"):
            st.session_state.current_log_filename = None
            st.session_state.chat_messages = []
            st.success("已成功开启新的对话")
            st.rerun()

        if interaction_mode == "多模态模式":
            model_display = st.selectbox("选择模型", list(VISIONMODAL_MAPPING.keys()), index=1, help="选择模型")
            model = VISIONMODAL_MAPPING[model_display]
        else:
            model_display = st.selectbox("选择模型", list(MODEL_MAPPING.keys()), index=1, help="选择模型")
            model = MODEL_MAPPING[model_display]

        if model not in REASON_MODELS:
            st.session_state.system_prompt = "You are a helpful assistant."

        ParameterConfiguration()

    if interaction_mode == "多模态模式":
        with st.expander("图片上传", expanded=False):
            uploaded_image = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_image:
                st.image(uploaded_image, caption="图片预览", use_container_width=True)

    # 在显示历史消息前添加动态计数器
    msg_counter = st.empty()
    msg_counter.markdown(f"""
    <div style='text-align: center; margin: 10px 0; font-size:14px;'>
        当前对话消息数：<span style='color: #ff4b4b; font-weight:bold;'>{len(st.session_state.chat_messages)}</span>/40
    </div>
    """, unsafe_allow_html=True)

    # 显示历史消息
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("在这里输入您的问题："):
        current_prompt = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(current_prompt)
        msg_counter.markdown(f"""
        <div style='text-align: center; margin: 10px 0; font-size:14px;'>
            当前对话消息数：<span style='color: #ff4b4b; font-weight:bold;'>{len(st.session_state.chat_messages)}</span>/40
        </div>
        """, unsafe_allow_html=True)

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.search_mode in SEARCH_METHODS:
            try:
                search = WebSearch(query=prompt, max_results=st.session_state.search_max_results)
                method = getattr(search, SEARCH_METHODS[st.session_state.search_mode])
                st.session_state.search_result = method()

                # 显示搜索结果
                with st.chat_message("assistant"):
                    st.markdown("🔍 搜索到以下相关信息：")
                    for i, result in enumerate(st.session_state.search_result):
                        st.markdown(f"{i + 1}. [{result['title']}]({result['href']})")
                        st.caption(result['body'][:min(len(result['body']), 100)] + "...")
            except Exception as e:
                st.error(f"没有检索到答案哦，错误信息:{e}")
                st.session_state.search_result = None

        # AI响应
        with st.chat_message("assistant"):
            try:
                messages = [{"role": "system", "content": get_system_prompt()}]
                base64_image = encode_image_to_base64(uploaded_image) if uploaded_image else None
                if base64_image:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    })
                else:
                    messages.append({"role": "user", "content": prompt})

                messages.extend([{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_messages])

                if st.session_state.stream:
                    reason_placeholder = st.empty()
                    message_placeholder = st.empty()
                    content = ""
                    reasoning_content = ""

                    for chunk in st.session_state.openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=st.session_state.temperature,
                            top_p=st.session_state.top_p,
                            presence_penalty=st.session_state.presence_penalty,
                            frequency_penalty=st.session_state.frequency_penalty,
                            max_tokens=st.session_state.max_tokens,
                            stream=True
                    ):
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if getattr(delta, 'reasoning_content', None):
                                reasoning_content += delta.reasoning_content
                                reason_placeholder.markdown(
                                    f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                                    f"🤔 {reasoning_content}</div>",
                                    unsafe_allow_html=True
                                )
                            if delta and delta.content is not None:
                                content += delta.content
                                message_placeholder.markdown(
                                    f"<div style='font-size:16px; margin-top:10px;'>{content}</div>",
                                    unsafe_allow_html=True
                                )
                    assistant_response = content
                else:
                    response = st.session_state.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                        presence_penalty=st.session_state.presence_penalty,
                        frequency_penalty=st.session_state.frequency_penalty,
                        max_tokens=st.session_state.max_tokens,
                        stream=False
                    )
                    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
                    assistant_response = response.choices[0].message.content

                    if reasoning_content:
                        st.markdown(
                            f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                            f"🤔 {reasoning_content}</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown(assistant_response)
                current_response = {"role": "assistant", "content": assistant_response}
                st.session_state.chat_messages.append(current_response)

                if len(st.session_state.chat_messages) > 40:
                    st.session_state.chat_messages = st.session_state.chat_messages[-40:]

                if st.session_state.current_user:
                    new_filename = st.session_state.log_manager.save_chat_log(
                        st.session_state.current_user,
                        [current_prompt, current_response],
                        log_filename=st.session_state.current_log_filename
                    )
                    if st.session_state.current_log_filename is None:
                        st.session_state.current_log_filename = new_filename

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


if __name__ == "__main__":
    main()
