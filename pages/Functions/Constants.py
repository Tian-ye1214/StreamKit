HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "文心一言-4.5": "ernie-4.5-turbo-128k",
    "豆包-1.6-thinking": "doubao-seed-1.6-thinking",
    "混元-T1(腾讯元宝)": "hunyuan-t1-latest",
    "Kimi-thinking": "kimi-thinking",
    "MiniMax-M1": "MiniMax-M1",
    "Claude-4-Thinking": "claude-sonnet-4-20250514-thinking",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-O3": "o3",
    "GPT-4.1": "gpt-4.1",
    "Gemini-2.5-pro-thinking": "gemini-2.5-pro-preview-06-05-thinking",
    "Grok4": "grok-4",
}

VISIONMODAL_MAPPING = {
    "Qwen2.5-VL-72B": "qwen2.5-vl-72b-instruct",
    "文心一言-4.5": "ernie-4.5-turbo-vl-32k",
    "智谱清言-4": "glm-4v-plus",
    "Gemini2.5-pro-thinking": "gemini-2.5-pro-preview-06-05-thinking",
    "ChatGPT-4o": "chatgpt-4o-latest",
    "GPT-4.1": "gpt-4.1",
    "GPT-O3": "o3",
    "Claude-4-Thinking": "claude-sonnet-4-20250514-thinking",
}

MAX_TOKEN_LIMIT = {
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "qwen3-235b-a22b": 32768,
    "qwen2.5-vl-72b-instruct": 16384,
    "glm-4-plus": 32768,
    "glm-4v-plus": 32768,
    "ernie-4.5-turbo-vl-32k": 32768,
    "ernie-4.5-turbo-128k": 131072,
    "doubao-seed-1.6-thinking": 32768,
    "hunyuan-t1-latest": 32768,
    "gemini-2.5-pro-preview-06-05-thinking": 200000,
    "chatgpt-4o-latest": 131072,
    "o3": 200000,
    "gpt-4.1": 131072,
    "grok-4": 200000,
    "claude-sonnet-4-20250514-thinking": 200000,
}

EMBEDDING_MODEL_MAPPING = {
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "BGE-M3": "Pro/BAAI/bge-m3",
}

SEARCH_METHODS = {
    "文本搜索": "text_search",
    "新闻搜索": "news_search",
    "图片搜索": "image_search",
    "视频搜索": "video_search"
}
