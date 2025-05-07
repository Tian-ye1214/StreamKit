MODEL_MAPPING = {
    "DeepSeek-V3-0324": "Pro/deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "Pro/deepseek-ai/DeepSeek-R1",
    "QwQ-32B": "Qwen/QwQ-32B",
    "R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct-128K",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
}

VISIONMODAL_MAPPING = {
    "Qwen2.5-VL-72B": "qwen2.5-vl-72b-instruct",
    "Qwen2.5-VL-7B": "qwen2.5-vl-7b-instruct",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-4o-mini": "gpt-4o-mini",
    "Gemini-2.0": "gemini-2.0-flash",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-4.1": "gpt-4.1",
    "O4-mini": "o4-mini",
    "Claude3.7": "claude-3-7-sonnet-20250219",
}

HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "通义千问3-30B": "qwen3-30b-a3b",
    "通义千问3-32B": "qwen3-32b",
    "通义千问-Max": "qwen-max-latest",
    "智谱清言-4": "glm-4-plus",
    "豆包": "Doubao-pro-32k",
    "GPT-4.1-nano": "gpt-4.1-nano",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude3.7": "claude-3-7-sonnet-20250219",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-2.0-Flash-exp": "gemini-2.0-flash-exp",
}

EMBEDDING_MODEL_MAPPING = {
    "BGE-M3": "Pro/BAAI/bge-m3",
    "BGE-large-中文": "BAAI/bge-large-zh-v1.5",
    "BGE-large-英文": "BAAI/bge-large-en-v1.5",
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
