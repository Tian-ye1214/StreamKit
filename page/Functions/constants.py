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
    "QVQ(支持多模态)": "Qwen/QVQ-72B-Preview",
    "Qwen2-VL-72B(支持多模态)": "Qwen/Qwen2-VL-72B-Instruct",
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

MULTIMODAL_MODELS = [
    "Qwen/QVQ-72B-Preview",
    "Qwen/Qwen2-VL-72B-Instruct"
]

SEARCH_METHODS = {
    "文本搜索": "text_search",
    "新闻搜索": "news_search",
    "图片搜索": "image_search",
    "视频搜索": "video_search"
}
