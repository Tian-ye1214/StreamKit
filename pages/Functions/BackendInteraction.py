import streamlit as st
from pages.Functions.ExtractFileContents import extract_text
from pages.Functions.Prompt import (
    generate_document_prompt,
    generate_search_prompt,
    generate_combined_prompt
)
from pages.Functions.UserLogManager import UserLogManager
from pages.Functions.ExtractFileContents import encode_image_to_base64
from pages.Functions.WebSearch import WebSearch
from pages.Functions.Constants import SEARCH_METHODS
from openai import OpenAI


class BackendInteractionLogic:
    def __init__(self):
        pass

    def initialize_session_state(self):
        """
        初始化各项参数，保存在session中
        """
        if "openai_client" not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key='sk-wxmqrirjoqrahuuyxbornwawplaobdlpxjefkzpfgiackdmu', base_url='https://api.siliconflow.cn/v1/')
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if len(st.session_state.chat_messages) > 40:
            st.session_state.chat_messages = st.session_state.chat_messages[-40:]

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

        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None

    def user_interaction(self):
        """
        用户注册/登录/登出
        """
        st.markdown("### 用户登录")
        username = st.text_input("请输入用户名", key="username_input")

        col1, col2 = st.columns([1, 1])
        with col1:
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
            with col2:
                    if st.button("退出登录"):
                        st.session_state.current_user = None
                        st.session_state.chat_messages = []
                        if 'current_log_filename' in st.session_state:
                            del st.session_state.current_log_filename
                        if 'delete_target' in st.session_state:
                            del st.session_state.delete_target
                        st.rerun()

            # 历史记录查询
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
            else:
                st.info("该用户暂无历史对话记录")

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

    def image_upload(self):
        with st.expander("图片上传", expanded=False):
            st.session_state.uploaded_image = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )
            if st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, caption="图片预览", use_container_width=True)

    def start_new_conversation(self):
        if st.button("开启新对话", help="开启新对话将清空当前对话记录"):
            st.session_state.current_log_filename = None
            st.session_state.chat_messages = []
            st.success("已成功开启新的对话")
            st.rerun()

    def parameter_configuration(self):
        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.1,
                                                         help="控制响应的随机性，值越高表示响应越随机")
                st.session_state.presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1,
                                                              help="正值会根据新主题惩罚模型，负值会使模型更倾向于重复内容")
                st.session_state.max_tokens = st.number_input("Max Tokens",
                                                              min_value=1,
                                                              max_value=8192,
                                                              value=4096,
                                                              help="生成文本的最大长度")

            with col2:
                st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1,
                                                   help="控制词汇选择的多样性")
                st.session_state.frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1,
                                                               help="正值会根据文本频率惩罚模型，负值鼓励重复")
                st.session_state.stream = st.toggle("流式输出", value=True,
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

    def get_system_prompt(self):
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

    def search_interaction(self):
        if st.session_state.search_mode in SEARCH_METHODS:
            try:
                search = WebSearch(query=st.session_state.prompt, max_results=st.session_state.search_max_results)
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

    def ai_generation(self):
        st.session_state.messages = [{"role": "system", "content": self.get_system_prompt()}]
        base64_image = encode_image_to_base64(st.session_state.uploaded_image) if st.session_state.get("uploaded_image",
                                                                                                       None) else None
        if base64_image:
            st.session_state.messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": st.session_state.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            })
        else:
            st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})

        st.session_state.messages.extend([{"role": m["role"], "content": m["content"]}
                                          for m in st.session_state.chat_messages])

        if st.session_state.stream:
            reason_placeholder = st.empty()
            message_placeholder = st.empty()
            content = ""
            reasoning_content = ""

            for chunk in st.session_state.openai_client.chat.completions.create(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
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
                model=st.session_state.model,
                messages=st.session_state.messages,
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

        copy_script = f"""
            <div id="copy-container-{id(assistant_response)}" style="display:inline;">
                <button onclick="copyToClipboard{id(assistant_response)}()" 
                        style="margin-left:10px; background:#f0f0f0; border:none; border-radius:3px; padding:2px 8px;"
                        title="复制内容">
                    📋
                </button>
                <div id="copy-content-{id(assistant_response)}" style="display:none; white-space: pre-wrap;">{assistant_response.lstrip()}</div>
            </div>
            <script>
                function copyToClipboard{id(assistant_response)}() {{
                    const content = document.getElementById('copy-content-{id(assistant_response)}').innerText;
                    navigator.clipboard.writeText(content);
                    const btn = event.target;
                    btn.innerHTML = '✅';
                    setTimeout(() => {{ btn.innerHTML = '📋'; }}, 500);
                }}
            </script>
            """
        st.components.v1.html(copy_script, height=30)

        current_response = {"role": "assistant", "content": assistant_response}
        st.session_state.chat_messages.append(current_response)

        if len(st.session_state.chat_messages) > 40:
            st.session_state.chat_messages = st.session_state.chat_messages[-40:]

        if st.session_state.current_user:
            new_filename = st.session_state.log_manager.save_chat_log(
                st.session_state.current_user,
                [st.session_state.current_prompt, current_response],
                log_filename=st.session_state.current_log_filename
            )
            if st.session_state.current_log_filename is None:
                st.session_state.current_log_filename = new_filename

    def user_input(self, prompt):
        msg_counter = st.empty()
        st.session_state.prompt = prompt
        st.session_state.current_prompt = {"role": "user", "content": st.session_state.prompt}
        st.session_state.chat_messages.append(st.session_state.current_prompt)
        msg_counter.markdown(f"""
        <div style='text-align: center; margin: 10px 0; font-size:14px;'>
            当前对话消息数：<span style='color: #ff4b4b; font-weight:bold;'>{len(st.session_state.chat_messages)}</span>/40
        </div>
        """, unsafe_allow_html=True)

        with st.chat_message("user"):
            st.markdown(st.session_state.prompt)
