HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "智谱GLM-4.6": "glm-4.6",
    "文心一言-4.5": "ernie-4.5-turbo-128k",
    "豆包-1.6-thinking": "doubao-seed-1.6-thinking",
    "Kimi-K2": "kimi-k2-0711-preview",
    "Claude-4.5-Sonnet": "claude-sonnet-4-5-20250929",
    "GPT-5.1": "gpt-5.1",
    "Gemini-3-pro-thinking": "gemini-3-pro-thinking",
    "Grok4.1": "grok-4-1-fast-reasoning",
    "LLaMA-4-Maverick": "llama-4-maverick-17b-128e-instruct",
}

VISIONMODAL_MAPPING = {
    "Qwen3-VL-235B": "qwen3-vl-235b-a22b-thinking",
    "文心一言-4.5": "ernie-4.5-turbo-vl-32k",
    "智谱GLM-4.5": "glm-4.5v",
    "Gemini-3-pro-thinking": "gemini-3-pro-thinking",
    "GPT-5.1": "gpt-5.1",
    "Claude-4.5-Sonnet": "claude-sonnet-4-5-20250929",
}

MAX_TOKEN_LIMIT = {
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "qwen3-235b-a22b": 32768,
    "qwen3-vl-235b-a22b-thinking": 32768,
    "glm-4.6": 32768,
    "glm-4.5v": 32768,
    "ernie-4.5-turbo-vl-32k": 32768,
    "ernie-4.5-turbo-128k": 131072,
    "doubao-seed-1.6-thinking": 32768,
    "gemini-3-pro-thinking": 200000,
    "gpt-5.1": 131072,
    "grok-4-1-fast-reasoning": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "llama-4-maverick-17b-128e-instruct": 200000,
}

EMBEDDING_MODEL_MAPPING = {
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "BGE-M3": "Pro/BAAI/bge-m3",
    "BCE-Base": "netease-youdao/bce-embedding-base_v1",
}

EMBEDDING_DIM = {
    "Qwen3-Embedding-8B": 4096,
    "Qwen3-Embedding-4B": 2048,
    "Qwen3-Embedding-0.6B": 1024,
    "BGE-M3": 1024,
    "BCE-Base": 768,
}

RERANKER_MODEL_MAPPING = {
    "Qwen3-Reranker-8B": "Qwen/Qwen3-Reranker-8B",
    "Qwen3-Reranker-4B": "Qwen/Qwen3-Reranker-4B",
    "Qwen3-Reranker-0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "BGE-reranker-v2": "BAAI/bge-reranker-v2-m3",
    "BCE-reranker": "netease-youdao/bce-reranker-base_v1",
}

SEARCH_METHODS = {
    "文本搜索": "text_search",
    "新闻搜索": "news_search",
    "图片搜索": "image_search",
    "视频搜索": "video_search"
}
