# 付费模型
MODEL_MAPPING = {
    "DeepSeek-v3": "Pro/deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "Pro/deepseek-ai/DeepSeek-R1",
    "QwQ": "Qwen/QwQ-32B",
    "R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct-128K",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
}

VISIONMODAL_MAPPING = {
    "QvQ-72B": "Qwen/QVQ-72B-Preview",
    "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen2-VL-7B": "Pro/Qwen/Qwen2-VL-7B-Instruct",
    "DeepSeek-VL2": "deepseek-ai/deepseek-vl2",
    "InternVL2": "OpenGVLab/InternVL2-26B",
    "Janus-pro-1B": "deepseek-ai/Janus-Pro-1B",
}

HIGHSPEED_MODEL_MAPPING = {
    "GPT-4o-mini": "gpt-4o-mini",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-2.0-Lite": "gemini-2.0-flash-lite-preview-02-05",
    "智谱清言": "glm-4-flashx",
    "百川智能": "Baichuan4-Air",
    "豆包": "Doubao-pro-32k",
}

EMBEDDING_MODEL_MAPPING = {
    "BGE-M3": "Pro/BAAI/bge-m3",
    "BGE-large-中文": "BAAI/bge-large-zh-v1.5",
    "BGE-large-英文": "BAAI/bge-large-en-v1.5",
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
