import streamlit as st
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pages.Functions.ExtractFileContents import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from pages.Functions.UserLogManager import KnowledgeBaseManager
from openai import OpenAI
import os
import re


kb_manager = KnowledgeBaseManager()

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

# 5. 加载rerank模型函数
def load_rerank_model():
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
    model.eval()
    model.to(DEVICE)
    return tokenizer, model

# 6. 使用rerank模型重排序
def rerank_results(query, results, tokenizer, model, top_k):
    if not results:
        return []
    
    # 准备输入对
    pairs = [[query, result["text"]] for result in results]
    
    # 使用模型计算分数
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()

    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]


def initialization():
    if "Client" not in st.session_state:
        st.session_state.Client = OpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))
    if 'rag_text' not in st.session_state:
        st.session_state.rag_text = []
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None


def RAG_base():
    with st.sidebar:
        st.header("用户管理")
        username = st.text_input("用户名", value="")
        user_col1, user_col2 = st.columns([1, 1])
        with user_col1:
            if st.button("登录/注册"):
                if username.strip() == "":
                    st.error("用户名不能为空")
                else:
                    if not re.match("^[A-Za-z0-9\u4e00-\u9fff]+$", username):
                        st.error("用户名只能包含中文、字母和数字")
                    else:
                        st.session_state.current_user = username
                        kb_manager.user_register(st.session_state.current_user)
                        st.success(f"欢迎, {st.session_state.current_user}!")

        st.subheader("我的知识库")
        if st.session_state.current_user is None:
            st.info("请先登录")
        else:
            knowledge_bases = kb_manager.list_knowledge_bases(st.session_state.current_user)
            if not knowledge_bases:
                st.info("您还没有上传任何知识库")
            else:
                for kb in knowledge_bases:
                    with st.expander(f"知识库: {kb['file_id']} - {kb['chunk_count']}段 - {kb['created_time']}"):
                        st.write(f"摘要: {kb['summary']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"删除知识库 {kb['file_id']}", key=f"delete_{kb['file_id']}"):
                                if kb_manager.delete_knowledge_base(st.session_state.current_user, kb['file_id']):
                                    st.success(f"知识库 {kb['file_id']} 已删除")
                                    st.rerun()
                                else:
                                    st.error(f"删除知识库 {kb['file_id']} 失败")

                        with col2:
                            if st.button(f"下载知识库 {kb['file_id']}", key=f"download_{kb['file_id']}"):
                                zip_buffer = kb_manager.download_knowledge_base(st.session_state.current_user, kb['file_id'])
                                if zip_buffer:
                                    st.download_button(
                                        label=f"点击下载知识库 {kb['file_id']}",
                                        data=zip_buffer,
                                        file_name=f"knowledge_{kb['file_id']}.zip",
                                        mime="application/zip",
                                        key=f"download_btn_{kb['file_id']}"
                                    )
                                else:
                                    st.error(f"下载知识库 {kb['file_id']} 失败")

                if st.button("下载所有知识库"):
                    zip_buffer = kb_manager.download_all_knowledge_bases(st.session_state.current_user)
                    if zip_buffer:
                        st.download_button(
                            label="点击下载所有知识库",
                            data=zip_buffer,
                            file_name=f"all_knowledge_bases_{st.session_state.current_user}.zip",
                            mime="application/zip",
                            key="download_all_btn"
                        )
                    else:
                        st.error("下载所有知识库失败")

        st.header("配置参数")
        chunk_size = st.number_input("文本块大小", min_value=64, max_value=8192, value=1024, step=64)
        overlap = st.number_input("重叠字符数", min_value=0, max_value=chunk_size//2, value=128, step=64)
        special_chars = st.text_input("特殊分隔符（用逗号分隔）", value="")
        special_chars = [char.strip() for char in special_chars.split(",")] if special_chars else None
        top_k = st.number_input("返回最相似的段落数", min_value=1, max_value=20, value=5, step=1)
        use_rerank = st.checkbox("启用Rerank重排序", value=False)
        if use_rerank:
            st.info("将使用BAAI/bge-reranker-v2-m3模型对检索结果进行重排序")

    col1, col2 = st.columns(2)

    if not st.session_state.current_user:
        st.warning("请先登录")
    
    with col2:
        st.header("知识库构建")
        if st.session_state.current_user:
            uploaded_file = st.file_uploader("上传文档", type=["pdf", "docx", "txt", "csv"])
            if uploaded_file is not None:
                file_id = file_hash(uploaded_file.read())
                uploaded_file.seek(0)  # 重置文件指针

                existing_kb = kb_manager.get_knowledge_base(st.session_state.current_user, file_id)
                if existing_kb:
                    st.success("该文档已存在知识库，无需重复构建。")
                    st.write(f"共 {len(existing_kb)} 段，已加载。")
                else:
                    text = extract_text(uploaded_file)
                    chunks = split_text(text, chunk_size=chunk_size, special_chars=special_chars, overlap=overlap)
                    st.write(f"文档被切割为 {len(chunks)} 段。")

                    with st.spinner("加载模型中..."):
                        tokenizer, embed_model = load_model()

                    with st.spinner("正在生成 embedding..."):
                        embeddings = []
                        batch_size = 8
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i:i + batch_size]
                            emb = hf_embed(batch, tokenizer, embed_model)
                            embeddings.extend(emb.tolist())

                    data = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
                    kb_manager.save_knowledge_base(st.session_state.current_user, file_id, data)
                    st.success(f"知识库已构建，共 {len(data)} 段。")
                    st.rerun()

    with col1:
        st.header("相似度检索")
        if st.session_state.current_user:
            if user_input := st.chat_input("在这里输入您的问题："):
                knowledge_bases = kb_manager.list_knowledge_bases(st.session_state.current_user)
                
                if not knowledge_bases:
                    st.warning("请先上传文档构建知识库！")
                else:
                    with st.spinner("加载模型中..."):
                        tokenizer, embed_model = load_model()
                    
                    # 计算用户输入的embedding
                    query_embedding = hf_embed([user_input], tokenizer, embed_model)[0]
                    
                    # 存储所有相似度结果
                    all_results = []
                    
                    # 遍历所有知识库文件
                    for kb in knowledge_bases:
                        data = kb_manager.get_knowledge_base(st.session_state.current_user, kb['file_id'])
                        
                        # 计算相似度
                        embeddings = np.array([item["embedding"] for item in data])
                        similarities = cosine_similarity([query_embedding], embeddings)[0]
                        
                        # 将结果添加到列表
                        for i, similarity in enumerate(similarities):
                            all_results.append({
                                "text": data[i]["text"],
                                "similarity": similarity,
                                "source": f"knowledge_{kb['file_id']}.json"
                            })
                    
                    # 按相似度排序并获取top-k
                    all_results.sort(key=lambda x: x["similarity"], reverse=True)
                    top_results = all_results[:top_k]

                    if use_rerank:
                        with st.spinner("正在使用Rerank模型重排序..."):
                            rerank_tokenizer, rerank_model = load_rerank_model()
                            candidate_results = all_results[:min(top_k * 4, len(all_results))]
                            top_results = rerank_results(user_input, candidate_results, rerank_tokenizer, rerank_model, top_k)
                    
                    # 显示结果
                    st.subheader(f"最相似的 {top_k} 个段落：")
                    print(top_results)
                    for i, result in enumerate(top_results, 1):
                        if use_rerank:
                            with st.expander(f"Rerank分数: {result['rerank_score']:.4f} - 来源: {result['source']}"):
                                st.write(result["text"])
                        else:
                            with st.expander(f"相似度: {result['similarity']:.4f} - 来源: {result['source']}"):
                                st.write(result["text"])

def rag_with_ai():
    pass

def main():
    initialization()
    st.title("文档上传与Embedding知识库构建")
    tab1, tab2 = st.tabs(['RAG Base', 'RAG With AI'])
    with tab1:
        RAG_base()
    with tab2:
        rag_with_ai()


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'RAG'
current_page = 'RAG'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
main()
