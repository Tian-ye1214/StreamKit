import os
import streamlit as st
from openai import OpenAI
from pages.Functions.ExtractFileContents import extract_text, encode_image_to_base64
from pages.Functions.UserLogManager import UserLogManager
from pages.Functions.Constants import MULTIMODAL_MAPPING
from pages.Functions.Prompt import (
    generate_document_prompt,
    generate_search_prompt,
    generate_combined_prompt
)

def initialize_session_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = 'sk-wxmqrirjoqrahuuyxbornwawplaobdlpxjefkzpfgiackdmu'
    if "base_url" not in st.session_state:
        st.session_state.base_url = os.getenv('OPENAI_BASE_URL', "https://api.siliconflow.cn/v1/")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if len(st.session_state.chat_messages) > 40:
        st.session_state.chat_messages = st.session_state.chat_messages[-40:]

    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = """You are a helpful assistant"""

    if "file_content" not in st.session_state:
        st.session_state.file_content = None

    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "log_manager" not in st.session_state:
        st.session_state.log_manager = UserLogManager()

    if "current_log_filename" not in st.session_state:
        st.session_state.current_log_filename = None

    if "search_mode" not in st.session_state:
        st.session_state.search_mode = None
    if "search_result" not in st.session_state:
        st.session_state.search_result = None

def main():
    initialize_session_state()

    st.set_page_config(layout="wide")
    st.title("Multi-modal Chat")

    
    with st.sidebar:
        st.session_state.openai_client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)
        st.subheader("选择模型")
        model = st.selectbox("模型", list(MULTIMODAL_MAPPING.keys()))

        
        model = MULTIMODAL_MAPPING[model]

        if st.button("开启新对话"):
                st.session_state.current_log_filename = None
                st.session_state.chat_messages = []
                st.success("已成功开启新的对话")
                st.rerun()

        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                                        help="控制响应的随机性，值越高表示响应越随机")
                presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1,
                                                help="正值会根据新主题惩罚模型，负值会使模型更倾向于重复内容")
                max_tokens = st.number_input("Max Tokens",
                                                min_value=1,
                                                max_value=8192,
                                                value=4096,
                                                help="生成文本的最大长度")

            with col2:
                top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1,
                                    help="控制词汇选择的多样性")
                frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1,
                                                help="正值会根据文本频率惩罚模型，负值鼓励重复")
                stream = st.toggle("流式输出", value=True,
                                    help="启用流式输出可以实时看到生成结果")

        with st.expander("Prompt设置", expanded=False):
            system_prompt = st.text_area("System Prompt",
                                            value=st.session_state.system_prompt,
                                            help="设置AI助手的角色和行为")
            if st.button("更新System Prompt"):
                st.session_state.system_prompt = system_prompt
                st.success("System Prompt已更新")

        with st.expander("文件上传", expanded=False):
            uploaded_file = st.file_uploader(
                "上传文件(支持PDF、Word、TxT、CSV)",
                type=["pdf", "docx", "txt", "csv"],
                accept_multiple_files=False
            )

            if uploaded_file:
                try:
                    file_content = extract_text(uploaded_file)
                    if file_content:
                        st.session_state.file_content = file_content
                        st.success("文件上传成功！")
                        st.text_area("文件内容预览",
                                        value=file_content[:300] + "...",
                                        height=150)
                except Exception as e:
                    st.error(f"文件处理失败: {str(e)}")

            if st.button("清除上传的文件"):
                st.session_state.file_content = None
                st.success("文件已清除")
                st.rerun()

        with st.expander("Temperature参数使用推荐", expanded=False):
            st.markdown("""
            | 场景 | 温度 |
            |------|------|
            | 代码生成/数学解题 | 0.0 |
            | 数据抽取/分析 | 1.0 |
            | 通用对话 | 1.3 |
            | 翻译 | 1.3 |
            | 创意写作/诗歌创作 | 1.5 |
            """)
    
    with st.expander("图片上传", expanded=False):
        uploaded_image = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_image:
            st.image(uploaded_image, caption="图片预览", use_container_width=True)

    if prompt := st.chat_input("在这里输入您的问题："):
        current_prompt = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(current_prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI响应
        with st.chat_message("assistant"):
            try:
                def get_system_prompt():
                    if st.session_state.file_content and st.session_state.search_result:
                        return generate_combined_prompt(
                            st.session_state.file_content,
                            st.session_state.search_result
                        )
                    if st.session_state.file_content:
                        return generate_document_prompt(st.session_state.file_content)
                    if st.session_state.search_result:
                        return generate_search_prompt(st.session_state.search_result)
                    return st.session_state.system_prompt

                messages = [{"role": "system", "content": get_system_prompt()}]

                if uploaded_image:
                    base64_image = encode_image_to_base64(uploaded_image)
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
                else:
                    messages.append({"role": "user", "content": prompt})

                messages.extend([{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_messages])

                if stream:
                    reason_placeholder = st.empty()
                    message_placeholder = st.empty()
                    content = ""
                    reasoning_content = ""

                    for chunk in st.session_state.openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            max_tokens=max_tokens,
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
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
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