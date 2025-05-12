HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "通义千问3-30B": "qwen3-30b-a3b",
    "通义千问3-32B": "qwen3-32b",
    "智谱清言-4": "glm-4-plus",
    "豆包": "Doubao-pro-32k",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "Grok-3-mini-fast": "grok-3-mini-fast-beta",
    "GPT-4.1": "gpt-4.1",
    "Gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "GPT-4.1-mini": "gpt-4.1-mini",
}

VISIONMODAL_MAPPING = {
    "Qwen2.5-VL-72B": "qwen2.5-vl-72b-instruct",
    "Qwen2.5-VL-7B": "qwen2.5-vl-7b-instruct",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-4.1": "gpt-4.1",
    "Claude3.7": "claude-3-7-sonnet-20250219",
}

MAX_TOKEN_LIMIT = {
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    "qwen3-235b-a22b": 64000,
    "qwen3-30b-a3b": 64000,
    "qwen3-32b": 64000,
    "glm-4-plus": 128000,
    "Doubao-pro-32k": 32000,
    "chatgpt-4o-latest": 128000,
    "grok-3-mini-fast-beta": 200000,
    "gpt-4.1": 128000,
    "gpt-4.1-mini": 128000,
    "qwen2.5-vl-72b-instruct": 16000,
    "qwen2.5-vl-7b-instruct": 16000,
    "claude-3-7-sonnet-20250219": 200000,
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
