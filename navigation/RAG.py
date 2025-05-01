import streamlit as st
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pages.Functions.ExtractFileContents import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from pages.Functions.UserLogManager import KnowledgeBaseManager
from pages.Functions.Prompt import rag_prompt
from openai import OpenAI
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING
import os
import re

kb_manager = KnowledgeBaseManager()


class RAG:
    def __init__(self, embedding_model_path, rerank_model_path):
        self.embedding_model_path = embedding_model_path
        self.rerank_model_path = rerank_model_path
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 定义 embedding 函数
    def hf_embed(self, texts: list[str], tokenizer, embed_model) -> np.ndarray:
        encoded_texts = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.DEVICE)
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
    def split_text(self, text, chunk_size=1024, special_chars=None, overlap=128):
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
    def file_hash(self, file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    # 4. 加载模型函数
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
        embed_model = AutoModel.from_pretrained(self.embedding_model_path, torch_dtype=torch.bfloat16)
        embed_model.eval()
        embed_model.to(self.DEVICE)
        return tokenizer, embed_model

    # 5. 加载rerank模型函数
    def load_rerank_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.rerank_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.rerank_model_path, torch_dtype=torch.bfloat16)
        model.eval()
        model.to(self.DEVICE)
        return tokenizer, model

    # 6. 使用rerank模型重排序
    def rerank_results(self, query, results, tokenizer, model, top_k):
        if not results:
            return []

        pairs = [[query, result["text"]] for result in results]

        # 使用模型计算分数
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
                self.DEVICE)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]

    def retrieval(self, user_input, tokenizer, embed_model, knowledge_bases):
        query_embedding = self.hf_embed([user_input], tokenizer, embed_model)[0]

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
        top_results = all_results[:st.session_state.top_k]

        if st.session_state.use_rerank:
            with st.spinner("正在使用Rerank模型重排序..."):
                rerank_tokenizer, rerank_model = self.load_rerank_model()
                candidate_results = all_results[:min(st.session_state.top_k * 4, len(all_results))]
                top_results = self.rerank_results(user_input, candidate_results, rerank_tokenizer,
                                                        rerank_model,
                                                        st.session_state.top_k)
        return top_results


def initialization(rag_system):
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1024
    if "overlap" not in st.session_state:
        st.session_state.overlap = 128
    if "special_chars" not in st.session_state:
        st.session_state.special_chars = ""
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "use_rerank" not in st.session_state:
        st.session_state.use_rerank = False
    if "Client" not in st.session_state:
        st.session_state.Client = OpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))
    if 'rag_text' not in st.session_state:
        st.session_state.rag_text = []
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "deepseek-chat"
    if 'embed_model' not in st.session_state:
        st.session_state.tokenizer, st.session_state.embed_model = rag_system.load_model()


def RAG_base(rag_system):
    col1, col2 = st.columns(2)

    if not st.session_state.current_user:
        st.info("请先登录")

    with col1:
        st.header("文段检索")
        if st.session_state.current_user:
            knowledge_bases = kb_manager.list_knowledge_bases(st.session_state.current_user)

            if not knowledge_bases:
                st.warning("请先上传文档构建知识库！")

            if user_input := st.chat_input("在这里输入您的问题：", key="rag_base_input"):
                top_results = rag_system.retrieval(user_input, st.session_state.tokenizer, st.session_state.embed_model, knowledge_bases)

                st.subheader(f"最相似的 {st.session_state.top_k} 个段落：")
                for i, result in enumerate(top_results, 1):
                    if st.session_state.use_rerank:
                        with st.expander(f"Rerank分数: {result['rerank_score']:.4f} - 来源: {result['source']}"):
                            st.write(result["text"])
                    else:
                        with st.expander(f"相似度: {result['similarity']:.4f} - 来源: {result['source']}"):
                            st.write(result["text"])

    with col2:
        st.header("知识库构建")
        if st.session_state.current_user:
            uploaded_files = st.file_uploader("上传文档(支持多个文件上传)", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)
            if uploaded_files:
                
                for uploaded_file in uploaded_files:
                    st.write(f"正在处理文件: {uploaded_file.name}")
                    file_id = rag_system.file_hash(uploaded_file.read())
                    uploaded_file.seek(0)

                    existing_kb = kb_manager.get_knowledge_base(st.session_state.current_user, file_id)
                    if existing_kb:
                        st.success(f"文件 {uploaded_file.name} 已存在知识库，无需重复构建。")
                        st.write(f"共 {len(existing_kb)} 段，已加载。")
                    else:
                        text = extract_text(uploaded_file)
                        chunks = rag_system.split_text(text, chunk_size=st.session_state.chunk_size,
                                                       special_chars=st.session_state.special_chars,
                                                       overlap=st.session_state.overlap)
                        st.write(f"文档 {uploaded_file.name} 被切割为 {len(chunks)} 段。")

                        with st.spinner(f"正在为 {uploaded_file.name} 生成 embedding..."):
                            embeddings = []
                            batch_size = 8
                            for i in range(0, len(chunks), batch_size):
                                batch = chunks[i:i + batch_size]
                                emb = rag_system.hf_embed(batch, st.session_state.tokenizer, st.session_state.embed_model)
                                embeddings.extend(emb.tolist())

                        data = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
                        kb_manager.save_knowledge_base(st.session_state.current_user, file_id, data)
                        st.success(f"文件 {uploaded_file.name} 的知识库已构建，共 {len(data)} 段。")
                
                st.rerun()


def rag_with_ai(rag_system):
    if not st.session_state.current_user:
        st.info("请先登录")
        return

    st.header("基于知识库的智能问答")
    knowledge_bases = kb_manager.list_knowledge_bases(st.session_state.current_user)

    if not knowledge_bases:
        st.warning("请先上传文档构建知识库！")
        return

    if user_input := st.chat_input("在这里输入您的问题：", key="rag_ai_input"):
        top_results = rag_system.retrieval(user_input, st.session_state.tokenizer, st.session_state.embed_model, knowledge_bases)

        context = "\n\n".join([f"参考内容 {i + 1}:\n{result['text']}" for i, result in enumerate(top_results)])
        message = rag_prompt(user_input, context)

        with st.spinner("正在生成回答..."):
            try:
                reason_placeholder = st.empty()
                message_placeholder = st.empty()
                content = ""
                reasoning_content = ""

                for chunk in st.session_state.Client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=message,
                        temperature=0.6,
                        max_tokens=8192,
                        stream=True
                ):
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if getattr(delta, 'reasoning_content', None):
                            reasoning_content += delta.reasoning_content
                            reason_placeholder.markdown(
                                f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                                f"🤔 {reasoning_content}</div>",
                                unsafe_allow_html=True
                            )
                        if delta and delta.content is not None:
                            content += delta.content
                            message_placeholder.markdown(
                                f"<div style='font-size:16px; margin-top:10px;'>{content}</div>",
                                unsafe_allow_html=True
                            )

                # 显示参考内容
                st.markdown("### 参考内容")
                for i, result in enumerate(top_results, 1):
                    if st.session_state.use_rerank:
                        with st.expander(f"Rerank分数: {result['rerank_score']:.4f} - 来源: {result['source']}"):
                            st.write(result["text"])
                    else:
                        with st.expander(f"相似度: {result['similarity']:.4f} - 来源: {result['source']}"):
                            st.write(result["text"])

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


def knowledge_base_management():
    st.subheader("我的知识库")
    if st.session_state.current_user is None:
        st.info("请先登录")
    else:
        knowledge_bases = kb_manager.list_knowledge_bases(st.session_state.current_user)
        if not knowledge_bases:
            st.info("您还没有上传任何知识库")
        else:
            if st.button("下载所有知识库", key=f"DownloadAllKnowledgeBase"):
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
            for kb in knowledge_bases:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.write(f"知识库: {kb['file_id']}")
                    st.write(f"创建时间: {kb['created_time']}")
                    st.write(f"段落数量: {kb['chunk_count']}段")
                
                with col2:
                    st.subheader("操作")
                    if st.button(f"下载", key=f"download_{kb['file_id']}"):
                        zip_buffer = kb_manager.download_knowledge_base(st.session_state.current_user, kb['file_id'])
                        if zip_buffer:
                            st.download_button(
                                label=f"点击下载",
                                data=zip_buffer,
                                file_name=f"knowledge_{kb['file_id']}.zip",
                                mime="application/zip",
                                key=f"download_btn_{kb['file_id']}"
                            )
                        else:
                            st.error(f"下载知识库 {kb['file_id']} 失败")

                    if st.button(f"删除", key=f"delete_{kb['file_id']}"):
                        if kb_manager.delete_knowledge_base(st.session_state.current_user, kb['file_id']):
                            st.success(f"知识库 {kb['file_id']} 已删除")
                            st.rerun()
                        else:
                            st.error(f"删除知识库 {kb['file_id']} 失败")

                with col3:
                    st.subheader("预览")
                    data = kb_manager.get_knowledge_base(st.session_state.current_user, kb['file_id'])
                    if data:
                        preview_text = data[0]["text"][:200] + "..." if len(data[0]["text"]) > 200 else data[0]["text"]
                        st.text_area("内容预览", preview_text, height=150)
                    else:
                        st.info("暂无预览内容")


def main():
    rag_system = RAG(embedding_model_path='G:/代码/ModelWeight/bge-m3',
                     rerank_model_path='G:/代码/ModelWeight/bge-reranker-v2-m3')
    initialization(rag_system)
    with st.sidebar:
        st.header("用户管理")
        username = st.text_input("用户名", value="")
        user_col1, user_col2 = st.columns([1, 1])
        with user_col1:
            if st.button("登录/注册", key="login"):
                if username.strip() == "":
                    st.error("用户名不能为空")
                else:
                    if not re.match("^[A-Za-z0-9\u4e00-\u9fff]+$", username):
                        st.error("用户名只能包含中文、字母和数字")
                    else:
                        st.session_state.current_user = username
                        kb_manager.user_register(st.session_state.current_user)
                        st.success(f"欢迎, {st.session_state.current_user}!")

        st.header("配置参数")
        model_names = list(HIGHSPEED_MODEL_MAPPING.keys())
        selected_model_name = st.selectbox(
            "选择模型",
            options=model_names,
            index=1
        )
        st.session_state.selected_model = HIGHSPEED_MODEL_MAPPING[selected_model_name]
        st.session_state.chunk_size = st.number_input("文本块大小", min_value=64, max_value=8192, value=1024, step=64)
        st.session_state.overlap = st.number_input("重叠字符数", min_value=0,
                                                   max_value=st.session_state.chunk_size // 2, value=128, step=64)
        st.session_state.special_chars = st.text_input("特殊分隔符（用逗号分隔）", value="")
        st.session_state.special_chars = [char.strip() for char in st.session_state.special_chars.split(
            ",")] if st.session_state.special_chars else None
        st.session_state.top_k = st.number_input("返回最相似的段落数", min_value=1, max_value=20, value=5, step=1)
        st.session_state.use_rerank = st.checkbox("启用Rerank重排序", value=False)
        if st.session_state.use_rerank:
            st.info("将使用BAAI/bge-reranker-v2-m3模型对检索结果进行重排序")
    st.title("文档上传与Embedding知识库构建")
    tab1, tab2, tab3 = st.tabs(['知识库管理', '知识库构建与展示', 'AI知识库问答'])
    with tab1:
        knowledge_base_management()
    with tab2:
        RAG_base(rag_system)
    with tab3:
        rag_with_ai(rag_system)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'RAG'
current_page = 'RAG'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
main()
