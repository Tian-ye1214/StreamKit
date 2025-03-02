import streamlit as st
import requests
import json


st.set_page_config(page_title="Dify API 交互", layout="wide")
st.title("Dify API 交互界面")
st.sidebar.header("输入参数")
keyword = st.sidebar.text_input("关键词", value="成都烤苕皮，美食，小吃")
title = st.sidebar.text_input("标题", value="成都乃至西南地区最好吃的烤苕皮")
audience = st.sidebar.text_input("目标受众", value="大众")
brands_to_avoid = st.sidebar.text_input("避免提及的品牌", value="无")
tone = st.sidebar.selectbox("语调", options=["专业的", "轻松的", "幽默的", "严肃的"], index=0)

url = 'https://api.dify.ai/v1/workflows/run'
api_key = "app-DD79epVFAh4Zl4Fg0DLH4MzY"
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}
result_container = st.container()

if st.sidebar.button("提交请求"):
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
        initial_message = st.info("程序正在运行，请耐心等待...")
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        image_placeholder = st.empty()
        
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
                                            status_placeholder.info(f"节点 '{node_title}' 开始处理")
                                        elif event_type == 'node_finished':
                                            node_title = json_data.get('data', {}).get('title', '未知节点')
                                            node_status = json_data.get('data', {}).get('status', '未知')
                                            outputs = json_data.get('data', {}).get('outputs', {})
                                            if outputs:
                                                if 'text' in outputs:
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
                                                try:
                                                    if isinstance(outputs, str):
                                                        try:
                                                            parsed_output = json.loads(outputs)
                                                            if 'output' in parsed_output:
                                                                text_content = parsed_output.get('output', '')
                                                                if text_content:
                                                                    full_response = text_content
                                                                    response_placeholder.markdown(full_response)

                                                        except json.JSONDecodeError:
                                                            full_response = outputs
                                                            response_placeholder.markdown(full_response)
                                                    else:
                                                        if 'text' in outputs:
                                                            text_content = outputs.get('text', '')
                                                            if text_content:
                                                                full_response = text_content
                                                                response_placeholder.markdown(full_response)
                                                except Exception as parse_error:
                                                    st.error(f"解析输出错误: {str(parse_error)}")
                                                    st.code(outputs)
                                            
                                            status_placeholder.success(f"工作流处理完成! 状态: {workflow_status}, 耗时: {elapsed_time:.2f}秒")
                                        elif event_type == 'message':
                                            message = json_data.get('data', {}).get('message', '')
                                            full_response += message
                                            response_placeholder.markdown(full_response)
                                    except json.JSONDecodeError:
                                        st.error(f"无法解析 JSON: {json_str}")
                else:
                    st.error(f"API 请求失败: {response.status_code}")
                    st.code(response.text)
        except Exception as e:
            st.error(f"发生错误: {str(e)}")


st.sidebar.markdown("---")
st.sidebar.markdown("[在线访问 Dify 应用](https://udify.app/workflow/IvKSTMs6nut4Y1FC)")

st.sidebar.markdown("---")
st.sidebar.subheader("使用说明")
st.sidebar.markdown("""
1. 在侧边栏填写所需参数
2. 点击"提交请求"按钮
3. 查看右侧的 API 响应结果
4. 状态信息会显示工作流的处理进度
""")
