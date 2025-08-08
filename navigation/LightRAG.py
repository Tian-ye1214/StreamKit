# -*- coding: utf-8 -*-
import streamlit as st
import random
import tempfile
import networkx as nx
from pyvis.network import Network
from pages.lightrag.llm.openai import openai_complete_if_cache
from pages.lightrag.llm.siliconcloud import siliconcloud_embedding
from pages.lightrag import LightRAG, QueryParam
from pages.lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from pages.lightrag.kg.shared_storage import initialize_pipeline_status
from pages.lightrag.rerank import custom_rerank
from pages.Functions.ExtractFileContents import extract_text
from pages.Functions.Constants import EMBEDDING_MODEL_MAPPING, HIGHSPEED_MODEL_MAPPING, EMBEDDING_DIM, \
    RERANKER_MODEL_MAPPING
import os
import asyncio
import logging
import logging.config

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.set_page_config(
    page_title="知识图谱检索",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize():
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = 'deepseek-chat'
    if "embedding_model" not in st.session_state:
        st.session_state.embed_model = 'Qwen/Qwen3-Embedding-8B'
    if "reranker_model" not in st.session_state:
        st.session_state.embed_model = 'Qwen/Qwen3-Reranker-8B'
    if "embed_dim" not in st.session_state:
        st.session_state.embed_dim = 4096
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if 'has_document' not in st.session_state:
        st.session_state.has_document = False
    if 'current_query_nodes' not in st.session_state:
        st.session_state.current_query_nodes = []
    if 'current_query_edges' not in st.session_state:
        st.session_state.current_query_edges = []


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


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    llm_model = st.session_state.get('llm_model')
    return await openai_complete_if_cache(
        llm_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.environ.get('ZhiZz_API_KEY'),
        base_url=os.environ.get('ZhiZz_URL'),
        **kwargs,
    )


async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
    reranker_model = st.session_state.get('reranker_model')
    return await custom_rerank(
        query=query,
        documents=documents,
        model=reranker_model,
        base_url="https://api.siliconflow.cn/v1/rerank",
        api_key=os.environ.get('SiliconFlow_API_KEY'),
        top_n=top_n or 10,
        **kwargs,
    )


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
            embedding_dim=st.session_state.embed_dim,
            max_token_size=8192,
            func=lambda texts: siliconcloud_embedding(
                texts,
                model=st.session_state.embed_model,
                max_token_size=8192,
                api_key=os.environ.get('SiliconFlow_API_KEY')
            ),
        ),
        rerank_model_func=my_rerank_func,
        addon_params={"language": "Simplified Chinese"},
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def get_user_working_dir(filename=None):
    base_dir = "user_workspaces"
    os.makedirs(base_dir, exist_ok=True)

    if not filename:
        raise ValueError("必须提供文件名来创建工作目录")

    dirname = os.path.splitext(filename)[0]
    return os.path.join(base_dir, dirname)


def process_grpah(G):
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

    st.info(f"📊 图谱统计: {len(G.nodes)} 个节点, {len(G.edges)} 个关系")
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f'graph_{random.randint(0, 999999)}.html')
    net.show(temp_path)
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
        
        // 创建可见节点集合（包括点击的节点和相连节点）
        var visibleNodeSet = new Set([clickedNode, ...connectedNodes]);
        
        // 批量准备更新数据
        var updateData = [];
        allNodes.forEach(function(nodeId) {
            updateData.push({
                id: nodeId,
                hidden: !visibleNodeSet.has(nodeId)
            });
        });
        
        // 一次性批量更新所有节点
        network.body.data.nodes.update(updateData);
        
        // 同时更新相关的边
        var allEdges = network.body.data.edges.getIds();
        var edgeUpdateData = [];
        allEdges.forEach(function(edgeId) {
            var edge = network.body.data.edges.get(edgeId);
            var shouldShow = visibleNodeSet.has(edge.from) && visibleNodeSet.has(edge.to);
            edgeUpdateData.push({
                id: edgeId,
                hidden: !shouldShow
            });
        });
        network.body.data.edges.update(edgeUpdateData);
        
    } else {
        // 点击空白区域时显示所有节点和边
        var allNodes = network.body.data.nodes.getIds();
        var allEdges = network.body.data.edges.getIds();
        
        // 批量更新所有节点为可见
        var nodeUpdateData = allNodes.map(function(nodeId) {
            return { id: nodeId, hidden: false };
        });
        network.body.data.nodes.update(nodeUpdateData);
        
        // 批量更新所有边为可见
        var edgeUpdateData = allEdges.map(function(edgeId) {
            return { id: edgeId, hidden: false };
        });
        network.body.data.edges.update(edgeUpdateData);
    }
});
</script>
"""
    html_content = html_content.replace('</body>', custom_js + '</body>')
    try:
        os.remove(temp_path)
    except Exception as e:
        pass
    return html_content


def display_knowledge_graph(working_dir, show_isolated=False):
    """显示知识图谱"""
    graph_path = os.path.join(working_dir, 'graph_chunk_entity_relation.graphml')
    if not os.path.exists(graph_path):
        return None
    try:
        G = nx.read_graphml(graph_path)
        if not show_isolated:
            G.remove_nodes_from(list(nx.isolates(G)))
        html_content = process_grpah(G)

    except Exception as e:
        st.warning(f"知识图谱显示失败: {str(e)}")
        html_content = None

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


def create_current_query_subgraph():
    """创建当前查询的子图谱并展示"""
    st.subheader("🔍 当前查询图谱")
    if not st.session_state.current_query_nodes and not st.session_state.current_query_edges:
        st.info("暂无查询图谱数据，请先进行查询")
        return

    try:
        G = nx.Graph()
        if st.session_state.current_query_nodes:
            for node_data in st.session_state.current_query_nodes:
                G.add_node(
                    str(node_data["id"]),
                    label=node_data["entity"],
                    entity_type=node_data["type"],
                    description=node_data["description"],
                    created_at=node_data["created_at"],
                    file_path=node_data["file_path"]
                )

        if st.session_state.current_query_edges:
            for edge_data in st.session_state.current_query_edges:
                edge_id = str(edge_data["id"])
                entity1_id = str(edge_data["entity1"])
                entity2_id = str(edge_data["entity2"])

                G.add_edge(
                    entity1_id,
                    entity2_id,
                    id=edge_id,
                    description=edge_data["description"],
                    created_at=edge_data["created_at"],
                    file_path=edge_data["file_path"]
                )

        if len(G.nodes) == 0:
            st.info("图谱中没有节点")
            return

        html_content = process_grpah(G)
        st.components.v1.html(html_content, height=800)

    except Exception as e:
        st.error(f"创建查询图谱时发生错误: {str(e)}")
        st.exception(e)


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


async def print_stream(stream, placeholder):
    content = ''
    async for chunk in stream:
        if chunk:
            content += chunk
            placeholder.markdown(content)
    return content


async def main():
    initialize()
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

        st.subheader("🤖 模型设置")
        model_display = st.selectbox("选择语言模型",
                                     list(HIGHSPEED_MODEL_MAPPING.keys()),
                                     index=0,
                                     help="选择模型",
                                     key="main_model_select")

        emb_model_display = st.selectbox("选择嵌入模型",
                                         list(EMBEDDING_MODEL_MAPPING.keys()),
                                         index=0,
                                         help="选择用于文本向量化的模型",
                                         key="embed_model_select")
        use_reranker = st.checkbox("是否启用重排序模型", value=False)
        if use_reranker:
            rerank_model_display = st.selectbox("选择重排序模型",
                                                list(RERANKER_MODEL_MAPPING.keys()),
                                                index=0,
                                                help="选择用于文本向量化的模型",
                                                key="reranker_model_select")
            st.session_state.reranker_model = RERANKER_MODEL_MAPPING[rerank_model_display]

        st.session_state.llm_model = HIGHSPEED_MODEL_MAPPING[model_display]
        st.session_state.embed_model = EMBEDDING_MODEL_MAPPING[emb_model_display]
        st.session_state.embed_dim = EMBEDDING_DIM[emb_model_display]

        if uploaded_file is not None:
            await _process_uploaded_file(uploaded_file)

        st.subheader("🔍 查询模式")
        query_mode = st.selectbox(
            "选择查询模式",
            options=["local", "global", "hybrid", "mix"],
            index=3,
            help="""- local: 局部模式，匹配最相似的实体关系及邻接实体关系
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
            top_k = st.slider("检索数量(top_k)", min_value=1, max_value=120, value=10,
                              help="在local模式下表示检索的实体数量，在global模式下表示检索的关系数量")
            chunk_top_k = st.slider("文档块检索数量", min_value=1, max_value=20, value=5,
                                    help="在重排序模式下，经过重排序后返回文档块数量")
            max_entity_tokens = st.slider("实体描述最大token数", min_value=5000, max_value=20000, value=8000,
                                                help="描述实体的最大token数量")
            max_relation_tokens = st.slider("关系描述最大token数", min_value=5000, max_value=8000,
                                                     value=10000, help="描述关系的最大token数量")
            max_total_tokens = st.slider("全局最大token数", min_value=10000, max_value=32768, value=30000,
                                                    help="总体上下文最大token数量")
            show_isolated_nodes = st.toggle("显示孤立节点", value=False, help="开启后将显示没有任何连接的节点")
        st.subheader("🗑️ 清除历史")
        if st.button("清除对话历史"):
            st.session_state.rag_messages = []
            st.success("已清除对话历史")

    if not st.session_state.has_document:
        st.warning("请先上传文档后再开始对话")
        return
    tab1, tab2 = st.tabs(['对话', '图谱全貌'])
    with tab1:
        col1, col2 = st.columns([4, 3])
        with col1:
            st.subheader("💬 对话")

            for message in st.session_state.rag_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("输入您的问题"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.current_query_nodes = []
                st.session_state.current_query_edges = []

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
                                chunk_top_k=chunk_top_k,
                                max_entity_tokens=max_entity_tokens,
                                max_relation_tokens=max_relation_tokens,
                                max_total_tokens=max_total_tokens,
                                conversation_history=st.session_state.rag_messages,
                                enable_rerank=use_reranker,
                                stream=True
                            )
                        )
                        placeholder = st.empty()
                        content = await print_stream(assistant_response, placeholder)
                        st.session_state.rag_messages.append({"role": "assistant", "content": content})
                    except Exception as e:
                        st.error(e)
                    finally:
                        if st.session_state.rag:
                            await st.session_state.rag.finalize_storages()

        with col2:
            create_current_query_subgraph()
    with tab2:
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
