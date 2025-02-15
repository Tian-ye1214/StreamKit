import streamlit as st
from pages.Functions.UserLogManager import UserLogManager


# 付费模型
MODEL_MAPPING = {
    "DeepSeek-v3": "Pro/deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "Pro/deepseek-ai/DeepSeek-R1",
    "R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "QWQ": "Qwen/QwQ-32B-Preview",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct-128K",
}
MULTIMODAL_MAPPING = {
    "QVQ-72B": "Qwen/QVQ-72B-Preview",
    "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen2-VL-7B": "Pro/Qwen/Qwen2-VL-7B-Instruct",
    "DeepSeek-VL2": "deepseek-ai/deepseek-vl2",
    "InternVL2": "OpenGVLab/InternVL2-26B",
    "Janus-pro-1B": "deepseek-ai/Janus-Pro-1B",
}

# 免费模型
FREE_MODEL_MAPPING = {
    "R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-Coder-7B": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "InternLM2.5-7B": "internlm/internlm2_5-7b-chat",
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "GLM-4-9B": "THUDM/glm-4-9b-chat",
    "Yi-1.5-9B": "01-ai/Yi-1.5-9B-Chat-16K",
    "Gemma2-9b": "google/gemma-2-9b-it",
}

REASON_MODELS = [
    "Pro/deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

SEARCH_METHODS = {
    "文本搜索": "text_search",
    "新闻搜索": "news_search",
    "图片搜索": "image_search",
    "视频搜索": "video_search"
}


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