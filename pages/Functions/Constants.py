HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "通义千问3-32B": "qwen3-32b",
    "文心一言-4.5": "ernie-4.5-turbo-128k",
    "智谱清言-4": "glm-4-plus",
    "豆包-1.5": "doubao-1.5-thinking-pro",
    "Claude-4": "claude-sonnet-4-20250514",
    "Claude-4-Thinking": "claude-sonnet-4-20250514-thinking",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-4.1": "gpt-4.1",
    "Gemini-2.5-pro": "gemini-2.5-pro-preview-06-05",
    "Gemini-2.5-pro-thinking": "gemini-2.5-pro-preview-06-05-thinking",
    "Grok3": "grok-3",
}

VISIONMODAL_MAPPING = {
    "Qwen2.5-VL-72B": "qwen2.5-vl-72b-instruct",
    "文心一言-4.5": "ernie-4.5-turbo-vl-32k",
    "智谱清言-4": "glm-4v-plus",
    "Gemini2.5-pro": "gemini-2.5-pro-preview-06-05",
    "Gemini2.5-pro-thinking": "gemini-2.5-pro-preview-06-05-thinking",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-4.1": "gpt-4.1",
    "Claude-4": "claude-sonnet-4-20250514",
    "Claude-4-Thinking": "claude-sonnet-4-20250514-thinking",
}

MAX_TOKEN_LIMIT = {
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "ernie-4.5-turbo-vl-32k": 32768,
    "ernie-4.5-turbo-128k": 131072,
    "doubao-1.5-thinking-pro": 32768,
    "gemini-2.5-pro-preview-06-05": 200000,
    "gemini-2.5-pro-preview-06-05-thinking": 200000,
    "LLaMa-4-Maverick": 32768,
    "qwen3-235b-a22b": 32768,
    "qwen3-32b": 32768,
    "glm-4-plus": 32768,
    "doubao-1.5-thinking-vision-pro": 32768,
    "chatgpt-4o-latest": 131072,
    "grok-3": 200000,
    "gpt-4.1": 131072,
    "qwen2.5-vl-72b-instruct": 16384,
    "claude-sonnet-4-20250514": 200000,
    "claude-sonnet-4-20250514-thinking": 200000,
}

EMBEDDING_MODEL_MAPPING = {
    "BGE-M3": "Pro/BAAI/bge-m3",
    "BGE-large-中文": "BAAI/bge-large-zh-v1.5",
    "BGE-large-英文": "BAAI/bge-large-en-v1.5",
}

SEARCH_METHODS = {
    "文本搜索": "text_search",
    "新闻搜索": "news_search",
    "图片搜索": "image_search",
    "视频搜索": "video_search"
}
