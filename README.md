<div align="center">
  <img src="static/logo.png" alt="StreamKit Logo" width="400" height="400">
  <h1>StreamKit -- AI Nexus</h1>
  <p>这是一个基于Streamlit开发的多功能Web应用平台，集成了多种AI和工具功能。</p>
</div>

~~具体套百家壳还差85家~~

## 🌟 主要功能

### 📜 言语洞澈(文本类应用)
- **AI对话平台**：支持多模态问答交互、解码参数调整、文件上传、网络搜索等功能
- **个人知识库(RAG)**：基于向量检索的智能问答系统，支持文档上传和知识管理
- **PDF固版翻译**：保持原PDF排版格式的智能翻译工具
- **论文分段润色**：学术论文智能润色和优化工具
- **小红书文案生成**：基于Dify工作流的智能文案生成

### 🔭 新域探微(研究类应用)
- **彝脉相承大模型**：基于Nuosu系列大模型的民族文化传承AI系统
- **古建筑图像生成**：SDXL驱动的古建筑风格图像生成工具
- **天眸预警**：天气实时预警和监测系统
- **知识图谱检索**：基于LightRAG的知识图谱构建和检索系统

### 🌌 融象观言(多模态类应用)
- **图像生成**：基于Nanobanana等先进模型的AI图像生成工具
- **视频生成**：支持Sora2模型文本到视频、图像到视频的生成功能
- **分割万物**：基于SAM3的智能语义分割工具，支持概念分割、点选和框选分割

## 🚀 快速开始

### 环境要求
- Python 3.12

### 安装步骤

1. 克隆项目并安装依赖：
```bash
git clone https://github.com/Tian-ye1214/StreamlitKit.git
pip install -r requirements.txt
```

2. 配置环境变量：
- 在项目根目录创建`.env`文件
- 配置必要的API密钥

3. 下载必要模型：
- 使用PDF固版翻译功能：
  - DocLayout模型：下载`doclayout_yolo_docstructbench_imgsz1024`模型并放入`pages/ModelCheckpoint`目录
  - 下载地址：[DocLayout](https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx/tree/main)
- 使用SAM3模型：
  - 下载sam3.pt并放入`pages/ModelCheckpoint/SAM3`目录
  - 下载地址：[SAM3](https://www.modelscope.cn/models/facebook/sam3/files)

4. 启动应用：
```bash
streamlit run main.py
```

## 📦 项目结构
```
.
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── .env                # 环境变量配置
├── navigation/         # 导航模块
├── pages/             # 功能页面
│   ├── Functions/  # 各种工具、常量配置
│   ├── SkySentry/  # 天眸预警项目知识库
│   ├── ModelCheckpoint/  # 模型检查点
│   └── SAM3/          # SAM3模型
└── user_logs/         # 用户日志
```

## 🔧 技术栈
- Streamlit：Web应用框架
- OpenAI：AI模型接口
- PyMuPDF：PDF处理
- 其他依赖见requirements.txt

## 🙏 致谢
感谢以下作者对代码的贡献：
- [@mwx66](https://github.com/mwx66)
- [@yanyunxi](https://github.com/yanyunxi)


