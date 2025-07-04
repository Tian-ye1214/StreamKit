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
from pages.Functions.Constants import SEARCH_METHODS, MAX_TOKEN_LIMIT
import re
from PIL import Image
import tiktoken
from pages.Functions.CallLLM import CallLLM


class BackendInteractionLogic:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        """统计文本的token数量"""
        return len(self.encoding.encode(text))

    def count_message_tokens(self, message):
        """统计消息的token数量"""
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, list):
                total_tokens = 0
                for item in content:
                    if item.get("type") == "text":
                        total_tokens += self.count_tokens(item.get("text", ""))
                return total_tokens
            return self.count_tokens(content)
        return self.count_tokens(str(message))

    async def initialize_session_state(self):
        """
        初始化各项参数，保存在session中
        """
        if "client" not in st.session_state:
            st.session_state.Client = CallLLM()
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if len(st.session_state.chat_messages) > 20:
            st.session_state.chat_messages = st.session_state.chat_messages[-20:]
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = (
                "You are a helpful assistant.You should reply to users in markdown format."
                "All mathematical formulas must be strictly enclosed between $...$")
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
        if 'total_tokens' not in st.session_state:
            st.session_state.total_tokens = 0

    async def user_interaction(self):
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
                    if not re.match("^[A-Za-z0-9\u4e00-\u9fff]+$", username):
                        st.error("用户名只能包含中文、字母和数字")
                    else:
                        st.session_state.current_user = username
                        if not await st.session_state.log_manager.check_user_exists(username):
                            st.success(f"欢迎 {'新用户'} ")
                            await st.session_state.log_manager.user_register(username)
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
            history_logs = await st.session_state.log_manager.get_user_history(st.session_state.current_user)
            if len(history_logs) > 0:
                st.markdown("### 历史对话")
                selected_log = st.selectbox("选择历史记录", history_logs)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("加载记录", help="读取并加载选中的对话记录"):
                        chat_log = await st.session_state.log_manager.load_chat_log(
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
                    json_data = await st.session_state.log_manager.get_log_filepath(
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
                st.info("暂无历史对话记录")

            if 'delete_target' in st.session_state:
                st.warning(f"确认要永久删除记录[{st.session_state.delete_target}]吗？该过程不可逆！")
                if st.button("确认删除", type="primary"):
                    try:
                        success = await st.session_state.log_manager.delete_chat_log(
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

    async def image_upload(self):
        with st.expander("图片上传", expanded=False):
            st.session_state.uploaded_image = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )
            if st.session_state.uploaded_image:
                image = Image.open(st.session_state.uploaded_image)
                width, height = image.size
                if width > 256 or height > 256:
                    scale = 256 / max(height, width)
                    new_h, new_w = int(height * scale), int(width * scale)
                    image = image.resize((new_w, new_h), Image.BILINEAR)
                st.image(image, caption="图片预览")

    async def start_new_conversation(self):
        if st.button("开启新对话", help="开启新对话将清空当前对话记录"):
            st.session_state.uploaded_image = None
            st.session_state.current_log_filename = None
            st.session_state.chat_messages = []
            st.success("已成功开启新的对话")
            st.rerun()

    async def parameter_configuration(self):
        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.1,
                                                         help="控制模型回答的多样性，值越高表示回复多样性越高")
                st.session_state.presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1,
                                                              help="控制回复主题的多样性性，值越高重复性越低")
                st.session_state.max_tokens = st.number_input("Max Tokens",
                                                              min_value=1,
                                                              max_value=32768,
                                                              value=8192,
                                                              help="生成文本的最大长度")

            with col2:
                st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.0, 0.1,
                                                   help="控制词汇选择的多样性,值越高表示潜在生成词汇越多样")
                st.session_state.frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1,
                                                               help="控制回复中相同词汇重复性，值越高重复性越低")
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
                    file_content = await extract_text(uploaded_file)
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

    async def get_system_prompt(self):
        if st.session_state.file_content and st.session_state.search_result and st.session_state.search_mode in SEARCH_METHODS:
            return generate_combined_prompt(
                st.session_state.file_content,
                st.session_state.search_result
            )
        if st.session_state.file_content:
            return generate_document_prompt(st.session_state.file_content)
        if st.session_state.search_result and st.session_state.search_mode in SEARCH_METHODS:
            return generate_search_prompt(st.session_state.search_result)
        return st.session_state.system_prompt

    async def search_interaction(self):
        if st.session_state.search_mode in SEARCH_METHODS:
            try:
                search = WebSearch(query=st.session_state.prompt, max_results=st.session_state.search_max_results)
                method = getattr(search, SEARCH_METHODS[st.session_state.search_mode])
                st.session_state.search_result = method()

                with st.chat_message("assistant"):
                    st.markdown("🔍 搜索到以下相关信息：")
                    for i, result in enumerate(st.session_state.search_result):
                        st.markdown(f"{i + 1}. [{result['title']}]({result['href']})")
                        st.caption(result['body'][:min(len(result['body']), 100)] + "...")
            except Exception as e:
                st.error(f"没有检索到答案哦，错误信息:{e}")
                st.session_state.search_result = None

    async def ai_generation(self, sections):
        st.session_state.messages = [{"role": "system", "content": await self.get_system_prompt()}]
        st.session_state.messages.extend([{"role": m["role"], "content": m["content"]}
                                          for m in st.session_state.chat_messages])
        st.session_state.chat_messages.append(st.session_state.current_prompt)
        base64_image = await encode_image_to_base64(st.session_state.uploaded_image) if st.session_state.get(
            "uploaded_image",
            None) else None
        if base64_image and sections == '视觉对话':
            st.session_state.messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": st.session_state.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            })
        else:
            st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})

        reason_placeholder = st.empty()
        message_placeholder = st.empty()
        model_parameter = {
            "model": st.session_state.model,
            "messages": st.session_state.messages,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p if st.session_state.top_p else None,
            "presence_penalty": st.session_state.presence_penalty,
            "frequency_penalty": st.session_state.frequency_penalty,
            "max_tokens": st.session_state.max_tokens
        }
        assistant_response = await st.session_state.Client.call(reason_placeholder, message_placeholder, st.session_state.stream, **model_parameter)

        input_tokens = sum(self.count_message_tokens(msg) for msg in st.session_state.messages)
        output_tokens = self.count_tokens(assistant_response)
        st.session_state.total_tokens = input_tokens + output_tokens
        st.session_state.token_counter.markdown(f"""
           <div style='text-align: center; margin: 10px 0; font-size:14px;'>
               当前Token数：<span style='color: #4b4bff; font-weight:bold;'>{st.session_state.total_tokens}</span>
        </div>
           """, unsafe_allow_html=True)
        if round(0.75 * MAX_TOKEN_LIMIT[st.session_state.model]) <= st.session_state.total_tokens < MAX_TOKEN_LIMIT[
            st.session_state.model]:
            st.warning(f"当前 {st.session_state.total_tokens} 个token将要超出模型限制。请减少输入的长度或调整模型。")
        elif st.session_state.total_tokens >= round(MAX_TOKEN_LIMIT[st.session_state.model]):
            st.error(f"当前 {st.session_state.total_tokens} 个token已经超出模型限制。请开启新的对话。")

        current_response = {"role": "assistant", "content": assistant_response}
        st.session_state.chat_messages.append(current_response)

        if len(st.session_state.chat_messages) > 20:
            st.session_state.chat_messages = st.session_state.chat_messages[-20:]

        if st.session_state.current_user:
            new_filename = await st.session_state.log_manager.save_chat_log(
                st.session_state.current_user,
                [st.session_state.current_prompt, current_response],
                log_filename=st.session_state.current_log_filename
            )
            if st.session_state.current_log_filename is None:
                st.session_state.current_log_filename = new_filename

    async def user_input(self, prompt):
        st.session_state.token_counter = st.empty()
        st.session_state.prompt = prompt
        st.session_state.current_prompt = {"role": "user", "content": st.session_state.prompt}
        with st.chat_message("user"):
            st.markdown(st.session_state.prompt)
