import streamlit as st
from pages.Functions.ExtractFileContents import extract_text
from pages.Functions.Prompt import (
    generate_document_prompt,
    generate_search_prompt,
    generate_combined_prompt
)
from pages.Functions.UserLogManager import UserLogManager


def initialize_session_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = 'sk-wxmqrirjoqrahuuyxbornwawplaobdlpxjefkzpfgiackdmu'
    if "base_url" not in st.session_state:
        st.session_state.base_url = 'https://api.siliconflow.cn/v1/'
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

    if 'interaction_mode' not in st.session_state:
        st.session_state.interaction_mode = "纯文本模式"


def UserInteraction():
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


def ParameterConfiguration():
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
