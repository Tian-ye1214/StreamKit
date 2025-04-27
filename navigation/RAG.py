import streamlit as st
import hashlib
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pages.Functions.ExtractFileContents import extract_text
from sklearn.metrics.pairwise import cosine_similarity

# 确保知识库文件夹存在
KNOWLEDGE_DIR = "user_knowledge"
if not os.path.exists(KNOWLEDGE_DIR):
    os.makedirs(KNOWLEDGE_DIR)

# 检查CUDA是否可用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 定义 embedding 函数
def hf_embed(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    encoded_texts = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = embed_model(
            input_ids=encoded_texts["input_ids"],
            attention_mask=encoded_texts["attention_mask"],
        )
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()

# 2. 文本切割函数
def split_text(text, chunk_size=1024, special_chars=None, overlap=128):
    if special_chars:
        # 先按特殊字符分割
        segments = []
        current_pos = 0
        for char in special_chars:
            parts = text[current_pos:].split(char)
            for i, part in enumerate(parts[:-1]):
                if part.strip():
                    segments.append(part.strip())
                current_pos += len(part) + len(char)
        if parts[-1].strip():
            segments.append(parts[-1].strip())

        chunks = []
        for segment in segments:
            if len(segment) > chunk_size:
                start = 0
                while start < len(segment):
                    end = min(start + chunk_size, len(segment))
                    chunks.append(segment[start:end])
                    start = end
                    if overlap > 0 and start < len(segment):
                        start = max(0, start - overlap)
            else:
                chunks.append(segment)
        return chunks
    else:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end
            if overlap > 0 and start < len(text):
                start = max(0, start - overlap)
        return chunks

# 3. 计算文件 hash
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# 4. 加载模型函数
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    embed_model = AutoModel.from_pretrained("BAAI/bge-m3")
    embed_model.eval()
    embed_model.to(DEVICE)
    return tokenizer, embed_model


def initialization():
    pass


def main():
    st.title("文档上传与Embedding知识库构建")
    st.sidebar.info(f"当前使用设备: {DEVICE}")

    with st.sidebar:
        st.header("配置参数")
        chunk_size = st.number_input("文本块大小", min_value=64, max_value=8192, value=1024, step=64)
        overlap = st.number_input("重叠字符数", min_value=0, max_value=chunk_size//2, value=128, step=64)
        special_chars = st.text_input("特殊分隔符（用逗号分隔）", value="")
        special_chars = [char.strip() for char in special_chars.split(",")] if special_chars else None
        top_k = st.number_input("返回最相似的段落数", min_value=1, max_value=20, value=5, step=1)

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("知识库构建")
        uploaded_file = st.file_uploader("上传文档", type=["pdf", "docx", "txt", "csv"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_id = file_hash(file_bytes)
            json_path = os.path.join(KNOWLEDGE_DIR, f"knowledge_{file_id}.json")

            if os.path.exists(json_path):
                st.success("该文档已存在知识库，无需重复构建。")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.write(f"共 {len(data)} 段，已加载。")
            else:
                text = extract_text(uploaded_file)
                chunks = split_text(text, chunk_size=chunk_size, special_chars=special_chars, overlap=overlap)
                st.write(f"文档被切割为 {len(chunks)} 段。")

                with st.spinner("加载模型中..."):
                    tokenizer, embed_model = load_model()

                with st.spinner("正在生成 embedding..."):
                    embeddings = []
                    batch_size = 1
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        emb = hf_embed(batch, tokenizer, embed_model)
                        embeddings.extend(emb.tolist())

                # 保存为 json
                data = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                st.success(f"知识库已构建，共 {len(data)} 段。")

    with col2:
        st.header("相似度检索")
        if user_input := st.chat_input("在这里输入您的问题："):
            # 获取所有知识库文件
            knowledge_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.startswith("knowledge_") and f.endswith(".json")]
            
            if not knowledge_files:
                st.warning("请先上传文档构建知识库！")
            else:
                with st.spinner("加载模型中..."):
                    tokenizer, embed_model = load_model()
                
                # 计算用户输入的embedding
                query_embedding = hf_embed([user_input], tokenizer, embed_model)[0]
                
                # 存储所有相似度结果
                all_results = []
                
                # 遍历所有知识库文件
                for json_file in knowledge_files:
                    with open(os.path.join(KNOWLEDGE_DIR, json_file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # 计算相似度
                    embeddings = np.array([item["embedding"] for item in data])
                    similarities = cosine_similarity([query_embedding], embeddings)[0]
                    
                    # 将结果添加到列表
                    for i, similarity in enumerate(similarities):
                        all_results.append({
                            "text": data[i]["text"],
                            "similarity": similarity,
                            "source": json_file
                        })
                
                # 按相似度排序并获取top-k
                all_results.sort(key=lambda x: x["similarity"], reverse=True)
                top_results = all_results[:top_k]
                
                # 显示结果
                st.subheader(f"最相似的 {top_k} 个段落：")
                for i, result in enumerate(top_results, 1):
                    with st.expander(f"相似度: {result['similarity']:.4f} - 来源: {result['source']}"):
                        st.write(result["text"])


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'RAG'
current_page = 'RAG'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
main()




