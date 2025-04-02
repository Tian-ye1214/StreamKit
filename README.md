# 这是一个Streamlit拼好网

啥东西都沾点的WebUI项目

~~距离套百家壳还差96家~~

## 主要功能

- AI交互：支持多模态、参数调整，文件上传、网络搜索等
- Dify交互：和Dify工作流交互
- 保持PDF排版格式的翻译
- 基于知识图谱的RAG
- SAM2.1语义分割

### 快速开始

 - pip install -r requirements.txt
 - 在.env中配置API_key
 - 下载doclayout_yolo_docstructbench_imgsz1024模型放入pages/ModelCheckpoint中：https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx/tree/main
 - 下载SAM2.1模型放入pages/SAM2_1/checkpoints中
 - streamlit run main.py
