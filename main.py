import os
import streamlit as st
from openai import OpenAI
from pages.Functions.ExtractFileContents import extract_text, encode_image_to_base64
from pages.Functions.UserLogManager import UserLogManager
from pages.Functions.Prompt import (
    generate_document_prompt,
    generate_search_prompt,
    generate_combined_prompt
)
from pages.Functions.Constants import MODEL_MAPPING, MULTIMODAL_MODELS, SEARCH_METHODS, REASON_MODELS

st.set_page_config(
    page_title="Chat With AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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
        st.session_state.system_prompt = ""

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

    st.markdown("""
    <h1 style='text-align: center;'>
        Chat With AI
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 用户登录")
        username = st.text_input("请输入用户名", key="username_input")

        if st.button("登录/注册"):
            if username.strip() == "":
                st.error("用户名不能为空")
            else:
                st.session_state.current_user = username
                if not st.session_state.log_manager.check_user_exists(username):
                    st.success(f"欢迎 {'新用户'} ")
                    st.session_state.log_manager.user_register(username)

                else:
                    st.success(f"欢迎 {'回来'} {username}！")

        if st.session_state.current_user:
            st.markdown(f"当前用户：**{st.session_state.current_user}**")
            history_logs = st.session_state.log_manager.get_user_history(st.session_state.current_user)

            if len(history_logs) > 0:
                st.markdown("### 历史对话")
                selected_log = st.selectbox("选择历史记录", history_logs)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("加载记录", help="读取并加载选中的对话记录"):
                        chat_log = st.session_state.log_manager.load_chat_log(
                            st.session_state.current_user,
                            selected_log
                        )
                        st.session_state.chat_messages = chat_log["messages"]
                        st.session_state.current_log_filename = selected_log + '.json'
                        st.rerun()

                with col2:
                    if st.button("删除记录", help="删除选中的对话记录"):
                        st.session_state.delete_target = selected_log
                
                with col3:
                    json_data = st.session_state.log_manager.get_log_filepath(
                        st.session_state.current_user,
                        selected_log + '.json'
                    )
                    with open(json_data, "rb") as f:
                        st.download_button(
                            label="下载记录",
                            data=f,
                            file_name=selected_log + '.json',
                            mime="application/json",
                            help="下载选中的对话记录到本地"
                        )

            if 'delete_target' in st.session_state:
                st.warning(f"确认要永久删除记录[{st.session_state.delete_target}]吗？该过程不可逆！")
                if st.button("确认删除", type="primary"):
                    try:
                        success = st.session_state.log_manager.delete_chat_log(
                            st.session_state.current_user,
                            st.session_state.delete_target + '.json'
                        )
                        if success:
                            st.success("记录已永久删除")
                            st.session_state.current_log_filename = None
                            st.session_state.chat_messages = []
                            del st.session_state.delete_target
                            st.rerun()
                        else:
                            st.error("删除失败：文件不存在")
                    except Exception as e:
                        st.error(f"删除失败：{str(e)}")
                if st.button("取消删除"):
                    del st.session_state.delete_target
                    st.rerun()

        else:
            st.info("该用户暂无历史对话记录")

        st.markdown("""
        <h3 style='text-align: center;'>
            模型配置
        </h3>
        """, unsafe_allow_html=True)
        st.session_state.openai_client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)

        model_display = st.selectbox("选择模型", list(MODEL_MAPPING.keys()), index=1, help="选择模型")
        model = MODEL_MAPPING[model_display]
        if model not in REASON_MODELS:
            st.session_state.system_prompt = "You are a helpful assistant."

        if st.button("开启新对话", help="开启新对话将清空当前对话记录"):
            st.session_state.current_log_filename = None
            st.session_state.chat_messages = []
            st.success("已成功开启新的对话")
            st.rerun()

        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.1,
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
                                     value=file_content[:200] + "...",
                                     height=150)
                except Exception as e:
                    st.error(f"文件处理失败: {str(e)}")

            if st.button("清除上传的文件"):
                st.session_state.file_content = None
                st.success("文件已清除")
                st.rerun()

        with st.expander("网络搜索", expanded=False):
            search_mode = st.selectbox(
                "选择搜索模式",
                ["关闭搜索", "文本搜索", "新闻搜索", "图片搜索", "视频搜索"],
                index=0
            )
            st.session_state.search_mode = None if search_mode == "关闭搜索" else search_mode

            if st.session_state.search_mode:
                st.session_state.search_max_results = st.number_input("最大结果数",
                                                                      min_value=1,
                                                                      max_value=5,
                                                                      value=3,
                                                                      help="设置最大返回的搜索结果数量")

        if model in MULTIMODAL_MODELS:
            with st.expander("图片上传", expanded=False):
                uploaded_image = st.file_uploader(
                    "上传图片",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_image:
                    st.image(uploaded_image, caption="图片预览", use_container_width=True)

        with st.expander("Temperature参数使用推荐", expanded=False):
            st.markdown("""
            | 场景 | 温度 |
            |------|------|
            | 代码生成/数学解题 | 0.0 |
            | 数据抽取/分析/推理 | 0.6 |
            | 通用对话 | 0.8 |
            | 翻译 | 1.0 |
            | 创意写作/诗歌创作 | 1.3 |
            """)

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
                from pages.Functions.WebSearch import WebSearch
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

                if model in MULTIMODAL_MODELS and uploaded_image:
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
