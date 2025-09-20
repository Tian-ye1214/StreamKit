HIGHSPEED_MODEL_MAPPING = {
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
    "通义千问3-235B": "qwen3-235b-a22b",
    "智谱GLM-4.5": "glm-4.5-x",
    "文心一言-4.5": "ernie-4.5-turbo-128k",
    "豆包-1.6-thinking": "doubao-seed-1.6-thinking",
    "混元-T1(腾讯元宝)": "hunyuan-t1-latest",
    "Kimi-K2": "kimi-k2-0711-preview",
    "MiniMax-M1": "MiniMax-M1",
    "Claude-4-Sonnet-Thinking": "claude-sonnet-4-20250514-thinking",
    "GPT-5": "gpt-5",
    "Gemini-2.5-pro-thinking": "gemini-2.5-pro-thinking",
    "Grok4": "grok-4",
    "LLaMA-4-Maverick": "llama-4-maverick-17b-128e-instruct",
}

VISIONMODAL_MAPPING = {
    "Qwen2.5-VL-72B": "qwen2.5-vl-72b-instruct",
    "文心一言-4.5": "ernie-4.5-turbo-vl-32k",
    "智谱GLM-4.5": "glm-4.5v",
    "Gemini2.5-pro-thinking": "gemini-2.5-pro-thinking",
    "GPT-5": "gpt-5",
    "Claude-4.1-Opus-Thinking": "claude-sonnet-4-20250514-thinking",
}

MAX_TOKEN_LIMIT = {
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "qwen3-235b-a22b": 32768,
    "qwen2.5-vl-72b-instruct": 16384,
    "glm-4.5-x": 32768,
    "glm-4.5v": 32768,
    "ernie-4.5-turbo-vl-32k": 32768,
    "ernie-4.5-turbo-128k": 131072,
    "doubao-seed-1.6-thinking": 32768,
    "hunyuan-t1-latest": 32768,
    "gemini-2.5-pro-thinking": 200000,
    "gpt-5": 131072,
    "grok-4": 200000,
    "claude-sonnet-4-20250514-thinking": 200000,
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
