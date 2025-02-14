# 新建常量文件存放模型映射
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
