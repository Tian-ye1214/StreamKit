# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json

st.session_state.clear()
st.set_page_config(page_title="Dify API 交互", layout="wide")
st.title("Dify API 交互界面")

st.sidebar.header("输入参数")
with st.expander("使用说明", expanded=False):
    st.markdown("""
        🌟 **小红书文案生成** 🌟
        
        ✅ 工作流进度跟踪<br>
        ✅ 多节点实时状态反馈<br>
        ✅ 企业级内容生成规范<br>

        📝 **创作指南**：
        1. 在侧边栏输入核心关键词
        2. 定制标题风格与受众定位
        3. 选择专业级语调参数
        4. 实时查看节点处理状态

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 15px;">
            🏆 典型应用场景：<br>
            • 自动生成产品营销文案<br>
            • 创建品牌合规内容<br>
            • 快速产出多语言宣传材料<br>
        </div>

        🔍 **状态追踪提示**：
        👁️ 实时查看节点处理进度<br>
        ⚡ 异常状态即时告警<br>
        📊 完整处理耗时统计
        """, unsafe_allow_html=True)

with st.sidebar:
    st.header("📝 创作参数")
    with st.expander("🔧 基础设置", expanded=True):
        keyword = st.text_input("核心关键词", value="成都烤苕皮，美食，小吃")
        title = st.text_input("文章标题", value="成都乃至西南地区最好吃的烤苕皮")
        audience = st.text_input("目标受众", value="大众")
    
    with st.expander("⚙️ 高级设置", expanded=False):
        brands_to_avoid = st.text_input("避免提及的品牌", value="无")
        tone = st.selectbox("内容语调", options=["专业的", "轻松的", "幽默的", "严肃的"], index=0)

url = 'https://api.dify.ai/v1/workflows/run'
api_key = "app-DD79epVFAh4Zl4Fg0DLH4MzY"
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}
result_container = st.container()

if st.sidebar.button("🚀 开始生成"):
    data = {
        "inputs": {
            "keyword": keyword,
            "title": title,
            "audience": audience,
            "brands_to_avoid": brands_to_avoid,
            "tone": tone,
        },
        "response_mode": "streaming",
        "user": "streamlit-user"
    }
    
    with result_container:
        status_cols = st.columns(3)
        with status_cols[0]:
            current_step = st.empty()
            current_step.markdown("### 📍 当前步骤\n\n等待启动...")
        with status_cols[1]:
            status_indicator = st.empty()
            status_indicator.markdown("### 🚦 处理状态\n\n准备就绪")
        with status_cols[2]:
            time_tracker = st.empty()
            time_tracker.markdown("### ⏱ 耗时统计\n\n0.00秒")
        
        initial_message = st.info("程序正在运行，请耐心等待...")
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        full_response = ""
        workflow_status = "处理中..."
        
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                if response.status_code == 200:
                    initial_message.empty()
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data:'):
                                json_str = line_text[5:].strip()
                                if json_str:
                                    try:
                                        json_data = json.loads(json_str)
                                        event_type = json_data.get('event', '')
                                        if event_type == 'workflow_started':
                                            status_placeholder.info("工作流已启动")
                                        elif event_type == 'node_started':
                                            node_title = json_data.get('data', {}).get('title', '未知节点')
                                            current_step.markdown(f"### 📍 当前步骤\n\n{node_title}")
                                            status_indicator.markdown("### 🚦 处理状态\n\n🔄 处理中...")
                                        elif event_type == 'node_finished':
                                            node_title = json_data.get('data', {}).get('title', '未知节点')
                                            node_status = json_data.get('data', {}).get('status', '未知')
                                            outputs = json_data.get('data', {}).get('outputs', {})
                                            if outputs:
                                                text_content = outputs.get('text', '')
                                                if text_content:
                                                    full_response = text_content
                                                    response_placeholder.markdown(full_response)
                                            
                                            status_placeholder.info(f"节点 '{node_title}' 处理完成，状态: {node_status}")
                                        elif event_type == 'workflow_finished':
                                            workflow_status = json_data.get('data', {}).get('status', '未知')
                                            elapsed_time = json_data.get('data', {}).get('elapsed_time', 0)
                                            outputs = json_data.get('data', {}).get('outputs', {})
                                            if outputs:
                                                text_content = outputs.get('text', '')
                                                if text_content:
                                                    full_response = text_content
                                                    response_placeholder.markdown(full_response)

                                            status_placeholder.success(f"工作流处理完成! 状态: {workflow_status}, 耗时: {elapsed_time:.2f}秒")
                                            time_tracker.markdown(f"### ⏱ 耗时统计\n\n{elapsed_time:.2f}秒")
                                            status_indicator.markdown("### 🚦 处理状态\n\n✅ 已完成")
                                    except json.JSONDecodeError:
                                        st.error(f"无法解析 JSON: {json_str}")
                else:
                    st.error(f"API 请求失败: {response.status_code}")
                    st.code(response.text)
        except Exception as e:
            st.error(f"发生错误: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("[在线访问 Dify 应用](https://udify.app/workflow/IvKSTMs6nut4Y1FC)")


