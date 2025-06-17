import json
import streamlit as st
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pages.Functions.ExtractFileContents import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from pages.Functions.UserLogManager import KnowledgeBaseManager
from pages.Functions.Prompt import rag_prompt, IntentRecognition
from openai import OpenAI
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING
import os
import re
import asyncio

kb_manager = KnowledgeBaseManager()


class RAG:
    def __init__(self, embedding_model_path, rerank_model_path):
        self.embedding_model_path = embedding_model_path
        self.rerank_model_path = rerank_model_path
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_results = []

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """获取每个序列最后一个有效token的embedding"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def format_instruction(self, instruction, query, doc, type='embedding'):
        if instruction is None:
            instruction = 'Given a search query, retrieve relevant passages that answer the query'
        if type == 'embedding':
            return f'Instruct: {instruction}\nQuery: {query}'
        else:
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc)
        return output

    # 1. 定义 embedding 函数
    async def hf_embed(self, texts: list[str], tokenizer, embed_model) -> np.ndarray:
        texts_with_instruct = [self.format_instruction(None, text, None, 'embedding') for text in texts]
        encoded_texts = tokenizer(
            texts_with_instruct, return_tensors="pt", padding=True, truncation=True, max_length=8192
        ).to(self.DEVICE)
        with torch.no_grad():
            outputs = embed_model(
                input_ids=encoded_texts["input_ids"],
                attention_mask=encoded_texts["attention_mask"],
            )
            embeddings = self.last_token_pool(outputs.last_hidden_state, encoded_texts["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        if embeddings.dtype == torch.bfloat16:
            return embeddings.detach().to(torch.float32).cpu().numpy()
        else:
            return embeddings.detach().cpu().numpy()

    # 2. 文本切割函数
    async def split_text(self, text, chunk_size=1024, special_chars=None, overlap=128):
        # 先按特殊字符分割
        if special_chars:
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
    async def file_hash(self, file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    # 4. 加载模型函数
    async def load_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
        embed_model = AutoModel.from_pretrained(self.embedding_model_path, torch_dtype=torch.bfloat16)
        embed_model.eval()
        embed_model.to(self.DEVICE)
        return tokenizer, embed_model

    # 5. 加载rerank模型函数
    async def load_rerank_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.rerank_model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(self.rerank_model_path, torch_dtype=torch.bfloat16)
        model.eval()
        model.to(self.DEVICE)
        # 初始化特殊token
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(self.suffix, add_special_tokens=False)
        
        return tokenizer, model

    def process_inputs(self, pairs, tokenizer):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=8192 - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=8192)
        for key in inputs:
            inputs[key] = inputs[key].to(self.DEVICE)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, model):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    # 6. 使用rerank模型重排序
    async def rerank_results(self, query, results, tokenizer, model, top_k):
        if not results:
            return []
        # 准备输入对
        pairs = [self.format_instruction(None, query, result["text"], type='rerank') for result in results]

        with torch.no_grad():
            inputs = self.process_inputs(pairs, tokenizer)
            scores = self.compute_logits(inputs, model)

        # 更新结果分数并排序
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]

    async def retrieval_from_kb(self, kb, query_embedding, all_results):
        data = await kb_manager.get_knowledge_base(st.session_state.current_user, kb['file_id'])
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

    async def retrieval(self, user_input, tokenizer, embed_model, knowledge_bases):
        embeddings = await self.hf_embed([user_input], tokenizer, embed_model)
        query_embedding = embeddings[0]

        tasks = [self.retrieval_from_kb(kb, query_embedding, self.all_results) for kb in knowledge_bases]
        await asyncio.gather(*tasks)

        # 按相似度排序并获取top-k
        self.all_results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = self.all_results[:st.session_state.top_k]

        if st.session_state.use_rerank:
            with st.spinner("正在使用Rerank模型重排序..."):
                rerank_tokenizer, rerank_model = await self.load_rerank_model()
                candidate_results = self.all_results[:min(st.session_state.top_k * 3, len(self.all_results))]
                top_results = await self.rerank_results(user_input, candidate_results, rerank_tokenizer,
                                                        rerank_model,
                                                        st.session_state.top_k)
        return top_results


async def initialization(rag_system):
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1024
    if "overlap" not in st.session_state:
        st.session_state.overlap = 128
    if "special_chars" not in st.session_state:
        st.session_state.special_chars = ""
    if "top_k" not in st.session_state:
        st.session_state.top_k = 2
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
        st.session_state.tokenizer, st.session_state.embed_model = await rag_system.load_embedding_model()


async def RAG_base(rag_system):
    col1, col2 = st.columns(2)

    if not st.session_state.current_user:
        st.info("请先登录")

    with col1:
        st.header("文段检索")
        if st.session_state.current_user:
            knowledge_bases = await kb_manager.list_knowledge_bases(st.session_state.current_user)

            if not knowledge_bases:
                st.warning("请先上传文档构建知识库！")

            if user_input := st.chat_input("在这里输入您的问题：", key="rag_base_input"):
                top_results = await rag_system.retrieval(user_input, st.session_state.tokenizer,
                                                         st.session_state.embed_model,
                                                         knowledge_bases)

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
            uploaded_files = st.file_uploader("上传文档(支持多个文件上传)", type=["pdf", "docx", "txt", "csv"],
                                              accept_multiple_files=True)
            if uploaded_files:
                tasks = [processing_file(rag_system, file) for file in uploaded_files]
                await asyncio.gather(*tasks)


async def processing_file(rag_system, uploaded_file):
    st.write(f"正在处理文件: {uploaded_file.name}")
    file_id = await rag_system.file_hash(uploaded_file.read())

    existing_kb = await kb_manager.get_knowledge_base(st.session_state.current_user, file_id)
    if not existing_kb:
        text = await extract_text(uploaded_file)
        chunks = await rag_system.split_text(text, chunk_size=st.session_state.chunk_size,
                                             special_chars=st.session_state.special_chars,
                                             overlap=st.session_state.overlap)
        st.write(f"文档 {uploaded_file.name} 被切割为 {len(chunks)} 段。")

        with st.spinner(f"正在为 {uploaded_file.name} 生成 embedding..."):
            embeddings = []
            batch_size = 8
            batch = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            task = [rag_system.hf_embed(text, st.session_state.tokenizer, st.session_state.embed_model) for text in
                    batch]
            result = await asyncio.gather(*task)
            for emb in result:
                embeddings.extend(emb.tolist())

        data = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
        await kb_manager.save_knowledge_base(st.session_state.current_user, file_id, data)
        st.success(f"文件 {uploaded_file.name} 的知识库已构建，共 {len(data)} 段。")


async def rag_with_ai(rag_system):
    if not st.session_state.current_user:
        st.info("请先登录")
        return

    st.header("基于知识库的智能问答")
    knowledge_bases = await kb_manager.list_knowledge_bases(st.session_state.current_user)

    if not knowledge_bases:
        st.warning("请先上传文档构建知识库！")
        return

    if user_input := st.chat_input("在这里输入您的问题：", key="rag_ai_input"):
        top_results = []
        with st.spinner('意图识别中：'):
            Intentmessage = IntentRecognition(user_input)
            response = st.session_state.Client.chat.completions.create(
                model='deepseek-chat',
                messages=Intentmessage,
                temperature=0.6,
                max_tokens=8192,
            )
            st.markdown(f'识别到意图:{response.choices[0].message.content}')
            user_intents = json.loads(response.choices[0].message.content)
        for user_intent in user_intents.values():
            top_results.extend(await rag_system.retrieval(user_intent, st.session_state.tokenizer,
                                                          st.session_state.embed_model, knowledge_bases))
        context = "\n".join([f"参考内容 {i + 1}:\n{result['text']}" for i, result in enumerate(top_results)])
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


async def knowledge_base_management():
    st.subheader("我的知识库")
    if st.session_state.current_user is None:
        st.info("请先登录")
    else:
        knowledge_bases = await kb_manager.list_knowledge_bases(st.session_state.current_user)
        if not knowledge_bases:
            st.info("您还没有上传任何知识库")
        else:
            if st.button("下载所有知识库", key=f"DownloadAllKnowledgeBase"):
                zip_buffer = await kb_manager.download_all_knowledge_bases(st.session_state.current_user)
                if zip_buffer:
                    st.download_button(
                        label="点击下载所有知识库",
                        data=zip_buffer.getvalue(),
                        file_name=f"all_knowledge_bases_{st.session_state.current_user}.zip",
                        mime="application/zip",
                        key="download_all_btn"
                    )
                else:
                    st.error("下载所有知识库失败")
            for kb in knowledge_bases:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.write(f"知识库: {kb['file_id']}")
                    st.write(f"创建时间: {kb['created_time']}")
                    st.write(f"段落数量: {kb['chunk_count']}段")

                with col2:
                    st.subheader("预览")
                    data = await kb_manager.get_knowledge_base(st.session_state.current_user, kb['file_id'])
                    if data:
                        preview_text = data[0]["text"][:200] + "..." if len(data[0]["text"]) > 200 else data[0]["text"]
                        st.text_area("内容预览", preview_text, height=150)
                    else:
                        st.info("暂无预览内容")

                with col3:
                    st.subheader("操作")
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        if st.button(f"下载", key=f"download_{kb['file_id']}"):
                            zip_buffer = await kb_manager.download_knowledge_base(st.session_state.current_user,
                                                                                  kb['file_id'])
                            if zip_buffer:
                                st.download_button(
                                    label=f"点击下载",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"knowledge_{kb['file_id']}.zip",
                                    mime="application/zip",
                                    key=f"download_btn_{kb['file_id']}"
                                )
                            else:
                                st.error(f"下载知识库 {kb['file_id']} 失败")

                    with col_2:
                        if st.button(f"删除", key=f"delete_{kb['file_id']}"):
                            if await kb_manager.delete_knowledge_base(st.session_state.current_user, kb['file_id']):
                                st.success(f"知识库 {kb['file_id']} 已删除")
                                st.rerun()
                            else:
                                st.error(f"删除知识库 {kb['file_id']} 失败")


async def main():
    st.markdown("""
    <h1 style='text-align: center;'>
        智识宝库 - 智能文档问答系统
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    rag_system = RAG(embedding_model_path='G:/代码/ModelWeight/Qwen3-embedding',
                     rerank_model_path='G:/代码/ModelWeight/Qwen3-rerank')
    await initialization(rag_system)

    with st.expander("📖 项目说明", expanded=False):
        st.markdown("""
        🌟 **智识宝库 - 智能文档问答系统** 🌟
        
        🧩 **系统功能**：
        ✅ 智能文档问答：基于知识库的智能问答系统<br>
        ✅ 知识库管理：支持文档上传、构建和管理<br>
        ✅ 多格式支持：支持PDF、Word、TXT、CSV等格式<br>
        ✅ 智能检索：基于语义相似度的文档检索<br>
        ✅ 重排序优化：支持Rerank模型优化检索结果<br>
        ✅ 多模型选择：支持多种大语言模型选择<br>

        📝 **主要模块**：
        1. 知识库管理
           - 上传和管理个人知识库
           - 支持文档预览和下载
           - 知识库的增删改查操作

        2. 知识库构建与展示
           - 文档上传和文本提取
           - 智能文本分割
           - 文档向量化存储
           - 相似度检索展示

        3. AI知识库问答
           - 基于知识库的智能问答
           - 多模型支持
           - 参考内容展示
           - 推理过程可视化

        ⚙️ **技术特点**：
        - 使用Qwen3-embedding模型进行文本向量化
        - 支持Qwen3-reranker进行结果重排序
        - 支持多用户知识库隔离

        💡 **使用建议**：
        - 建议先构建知识库后再进行问答
        - 可以根据需要调整文本块大小和重叠度
        - 启用Rerank可以提升检索质量
        - 选择合适的模型以获得最佳效果

        ⚠️ **注意事项**：
        - 请确保上传文档格式正确
        - 建议控制单个文档大小
        - 注意保护个人隐私信息
        - 定期备份重要知识库
        """, unsafe_allow_html=True)

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
                        await kb_manager.user_register(st.session_state.current_user)
                        st.success(f"欢迎, {st.session_state.current_user}!")

        st.header("配置参数")
        model_names = list(HIGHSPEED_MODEL_MAPPING.keys())
        selected_model_name = st.selectbox(
            "选择模型",
            options=model_names,
            index=0
        )
        st.session_state.selected_model = HIGHSPEED_MODEL_MAPPING[selected_model_name]
        st.session_state.chunk_size = st.number_input("文本块大小", min_value=64, max_value=8192, value=1024, step=64)
        st.session_state.overlap = st.number_input("重叠字符数", min_value=0,
                                                   max_value=st.session_state.chunk_size // 2, value=128, step=64)
        st.session_state.special_chars = st.text_input("特殊分隔符（用英文逗号分隔）", value="")
        st.session_state.special_chars = [char.strip() for char in st.session_state.special_chars.split(
            ",")] if st.session_state.special_chars else None
        st.session_state.top_k = st.number_input("返回最相似的段落数", min_value=1, max_value=20, value=2, step=1)
        st.session_state.use_rerank = st.checkbox("启用Rerank重排序", value=False)
        if st.session_state.use_rerank:
            st.info("将使用Qwen3-reranker模型对检索结果进行重排序")
    tab1, tab2, tab3 = st.tabs(['知识库管理', '知识库构建与展示', 'AI知识库问答'])
    with tab1:
        await knowledge_base_management()
    with tab2:
        await RAG_base(rag_system)
    with tab3:
        await rag_with_ai(rag_system)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'RAG'
current_page = 'RAG'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
asyncio.run(main())
