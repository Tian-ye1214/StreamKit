# -*- coding: utf-8 -*-
import streamlit as st
import random
import tempfile
import networkx as nx
from pyvis.network import Network
from pages.lightrag.llm.openai import openai_complete_if_cache, openai_embed
from pages.Functions.ExtractFileContents import extract_text
from pages.Functions.Constants import EMBEDDING_MODEL_MAPPING, HIGHSPEED_MODEL_MAPPING
import os
import asyncio
import logging
import logging.config
from pages.lightrag import LightRAG, QueryParam
from pages.lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from pages.lightrag.kg.shared_storage import initialize_pipeline_status

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        st.session_state.llm_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.environ.get('ZhiZz_API_KEY'),
        base_url=os.environ.get('ZhiZz_URL'),
        **kwargs,
    )


async def embedding_func(texts: list[str]):
    return await openai_embed(
        texts,
        model=st.session_state.embedding_model,
        api_key=os.environ.get('SiliconFlow_API_KEY'),
        base_url=os.environ.get('SiliconFlow_URL'),
    )


def get_user_working_dir(filename=None):
    base_dir = "user_workspaces"
    os.makedirs(base_dir, exist_ok=True)

    if not filename:
        raise ValueError("必须提供文件名来创建工作目录")

    dirname = os.path.splitext(filename)[0]
    return os.path.join(base_dir, dirname)


async def init_rag(filename=None):
    if not filename:
        raise ValueError("必须提供文件名来初始化RAG")

    working_dir = get_user_working_dir(filename)
    if os.path.exists(working_dir):
        st.info(f"检测到已存在的知识库：{os.path.basename(working_dir)}")
    else:
        os.makedirs(working_dir)
        st.success(f"创建新的知识库：{os.path.basename(working_dir)}")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096 if st.session_state.embedding_model == 'Qwen/Qwen3-Embedding-8B' else 1024,
            max_token_size=8192,
            func=embedding_func),
        addon_params={"language": "Simplified Chinese"},
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def load_knowledge_graph(graph_path, show_isolated=False):
    """加载并缓存知识图谱数据"""
    if not os.path.exists(graph_path):
        return None

    try:
        G = nx.read_graphml(graph_path)
        if not show_isolated:
            G.remove_nodes_from(list(nx.isolates(G)))

        net = Network(height="100vh", notebook=True, directed=False)
        net.from_nx(G)
        for node in net.nodes:
            node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            if "description" in node:
                node["title"] = node["description"]

        for edge in net.edges:
            if "description" in edge:
                edge["title"] = edge["description"]
            edge["width"] = 2

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'graph_{random.randint(0, 999999)}.html')
        net.show(temp_path)
        html_content = _load_temp_graph(temp_path)
        return html_content
    except Exception as e:
        st.warning(f"知识图谱显示失败: {str(e)}")
        return None


def _load_temp_graph(temp_path):
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    custom_js = """
    <script>
    // 等待网络图加载完成
    network.on("click", function(event) {
        if (event.nodes.length > 0) {
            var clickedNode = event.nodes[0];
            var allNodes = network.body.data.nodes.getIds();
            var connectedNodes = network.getConnectedNodes(clickedNode);

            // 隐藏所有节点
            allNodes.forEach(function(nodeId) {
                network.body.data.nodes.update({
                    id: nodeId,
                    hidden: true
                });
            });

            // 显示点击的节点及其相连的节点
            network.body.data.nodes.update({
                id: clickedNode,
                hidden: false
            });
            connectedNodes.forEach(function(nodeId) {
                network.body.data.nodes.update({
                    id: nodeId,
                    hidden: false
                });
            });

            // 更新网络图
            network.redraw();
        } else {
            // 点击空白区域时显示所有节点
            var allNodes = network.body.data.nodes.getIds();
            allNodes.forEach(function(nodeId) {
                network.body.data.nodes.update({
                    id: nodeId,
                    hidden: false
                });
            });
            network.redraw();
        }
    });
    </script>
    """
    return html_content.replace('</body>', custom_js + '</body>')


def display_knowledge_graph(working_dir, show_isolated=False):
    """显示知识图谱"""
    graph_path = os.path.join(working_dir, 'graph_chunk_entity_relation.graphml')
    html_content = load_knowledge_graph(graph_path, show_isolated)

    if html_content:
        with st.expander("知识图谱说明"):
            st.info("👇 这是从文档中提取的知识图谱，展示了文档中的实体及其关系。您可以：\n"
                    "- 拖动节点调整布局\n"
                    "- 滚轮缩放图谱\n"
                    "- 鼠标悬停在节点上查看详细信息\n"
                    "- 点击节点查看节点子图\n"
                    "- 点击空白部分显示全部节点\n")
        st.components.v1.html(html_content, height=800)
    else:
        st.info("暂无知图谱数据")


async def _process_uploaded_file(uploaded_file):
    with st.spinner('正在处理文档(首次处理文档会比较慢，请耐心等待哦)...'):
        current_file = st.session_state.get('current_file', None)
        if current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.rag = await init_rag(uploaded_file.name)
            st.session_state.has_document = False
            st.session_state.messages = []

        if not st.session_state.has_document:
            content = await extract_text(uploaded_file)
            if content:
                try:
                    await st.session_state.rag.ainsert(content)
                    st.success('文档处理完成！')
                    st.session_state.has_document = True
                except Exception as e:
                    st.error(f"处理文档时发生错误: {str(e)}")
                    st.session_state.has_document = False


async def main():
    st.title("LightRAG - 基于知识图谱的检索增强生成系统")
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **开启增强检索新时代** 🌟
        
        ✅ **源项目地址**：https://github.com/HKUDS/LightRAG

        🧩 **系统亮点**：
        ✅ 异步抽取知识图谱，获得更高速的体验<br>
        ✅ 利用AI构建细粒度知识图谱<br>
        ✅ 基于图谱内容进行增强式检索(RAG)<br>
        ✅ 图谱内容实时展示<br>

        🛠️ **操作指南**：
        1. 上传业务文档（合同/报告/手册等）
        2. 选择智能查询模式
        3. 探索动态生成的知识图谱
        4. 获取基于上下文的精准回答

        🌐 **图谱交互技巧**：
        🖱️ 右键拖动探索图谱全景<br>
        🔍 点击节点聚焦关联网络<br>
        🎚️ 滑动调节信息密度阈值
        """, unsafe_allow_html=True)

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "上传文档",
            type=['pdf', 'xlsx', 'xls', 'txt', 'doc', 'docx'],
            help="支持的格式：PDF、Excel文件(xlsx/xls)、文本文件(txt), Document文件(doc/docx)"
        )

        if uploaded_file is not None:
            await _process_uploaded_file(uploaded_file)

        if 'has_document' not in st.session_state:
            st.session_state.has_document = False

        st.subheader("🤖 模型设置")
        model_display = st.selectbox("选择模型",
                                     list(HIGHSPEED_MODEL_MAPPING.keys()),
                                     index=0,
                                     help="选择模型",
                                     key="main_model_select")

        emb_model_display = st.selectbox("选择模型",
                                         list(EMBEDDING_MODEL_MAPPING.keys()),
                                         index=0,
                                         help="选择用于文本向量化的模型",
                                         key="embed_model_select")

        new_llm_model = HIGHSPEED_MODEL_MAPPING[model_display]
        new_embedding_model = EMBEDDING_MODEL_MAPPING[emb_model_display]

        if (new_llm_model != st.session_state.get("llm_model") or
                new_embedding_model != st.session_state.get("embedding_model")):
            st.session_state.llm_model = new_llm_model
            st.session_state.embedding_model = new_embedding_model
            if "rag" in st.session_state and st.session_state.has_document:
                st.session_state.rag = await init_rag(st.session_state.current_file)
                st.info("更改模型成功！")
        st.subheader("🔍 查询模式")
        query_mode = st.selectbox(
            "选择查询模式",
            options=["naive", "local", "global", "hybrid", "mix"],
            index=4,
            help="""- naive: 朴素模式，直接匹配最相似的文本片段
                    - local: 局部模式，匹配最相似的实体关系及邻接实体关系
                    - global: 全局模式，匹配最相似的实体以及间接实体关系
                    - hybrid: local模式 + global模式
                    - mix: naive模式 + hybrid模式""",
            key="query_mode_select"
        )

        st.subheader("⚙️ 高级参数设置")
        with st.expander("展开高级参数设置"):
            only_need_context = st.toggle("仅返回上下文", help="开启后将只返回检索到的相关上下文，不进行LLM总结")
            only_need_prompt = st.toggle("仅返回提示词", help="开启后将只返回生成的提示词，不进行LLM回答")
            response_type = st.selectbox(
                "回答格式",
                options=["Multiple Paragraphs", "Single Paragraph", "Bullet Points"],
                help="选择AI回答的格式样式",
                key="response_type_select"
            )
            top_k = st.slider("检索数量(top_k)", min_value=10, max_value=100, value=60,
                              help="在local模式下表示检索的实体数量，在global模式下表示检索的关系数量")
            max_token_for_text_unit = st.slider("文本单元最大token数", min_value=1000, max_value=8000, value=4000,
                                                help="原始文本块的最大token数量")
            max_token_for_global_context = st.slider("全局上下文最大token数", min_value=1000, max_value=8000,
                                                     value=4000, help="关系描述的最大token数量")
            max_token_for_local_context = st.slider("局部上下文最大token数", min_value=1000, max_value=8000, value=4000,
                                                    help="实体描述的最大token数量")
            show_isolated_nodes = st.toggle("显示孤立节点", value=False, help="开启后将显示没有任何连接的节点")
        st.subheader("🗑️ 清除历史")
        if st.button("清除对话历史"):
            st.session_state.rag_messages = []
            st.success("已清除对话历史")

    if not st.session_state.has_document:
        st.warning("请先上传文档后再开始对话")
        return
    col1, col2 = st.columns([4, 3])
    with col1:
        st.subheader("💬 对话")
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("输入您的问题"):
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                try:
                    assistant_response = await st.session_state.rag.aquery(
                        query=prompt,
                        param=QueryParam(
                            mode=query_mode,
                            only_need_context=only_need_context,
                            only_need_prompt=only_need_prompt,
                            response_type=response_type,
                            top_k=top_k,
                            max_token_for_text_unit=max_token_for_text_unit,
                            max_token_for_global_context=max_token_for_global_context,
                            max_token_for_local_context=max_token_for_local_context,
                        )
                    )
                    st.markdown(assistant_response)
                    st.session_state.rag_messages.append({"role": "assistant", "content": assistant_response})

                    if len(st.session_state.rag_messages) > 20:
                        st.session_state.rag_messages = st.session_state.rag_messages[-20:]

                except Exception as outer_e:
                    st.error(f"处理过程发生错误: {str(outer_e)}")
                finally:
                    await st.session_state.rag.finalize_storages()

    with col2:
        st.subheader("📊 知识图谱")
        if not st.session_state.has_document:
            st.warning("请先上传文档以生成知识图谱")
        else:
            display_knowledge_graph(st.session_state.rag.working_dir, show_isolated_nodes)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'lightRAG'
current_page = 'lightRAG'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
configure_logging()
asyncio.run(main())
