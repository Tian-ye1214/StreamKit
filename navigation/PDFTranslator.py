# -*- coding: utf-8 -*-
import streamlit as st
from pdf2zh import translate_stream
import os
import fitz
import numpy as np
from pdf2zh.doclayout import OnnxModel


def translate_pdf(file, lang_in="en", lang_out="zh", service="google", thread=4, api_key=None, base_url=None,
                  model=None):
    """使用pdf2zh翻译PDF文件"""
    try:
        pdf_bytes = file.getvalue()

        if service == "openai" and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
            if model:
                os.environ["OPENAI_MODEL"] = model

        stream_mono, stream_dual = translate_stream(
            stream=pdf_bytes,
            lang_in=lang_in,
            lang_out=lang_out,
            service=service,
            thread=thread,
            model=OnnxModel('pages/ModelCheckpoint/doclayout_yolo_docstructbench_imgsz1024.onnx'),
        )
        return stream_mono, stream_dual
    except Exception as e:
        st.error(f"翻译过程出错：{str(e)}")
        return None, None


def display_pdf(file, prefix=""):
    """显示PDF文件
    Args:
        file: PDF文件字节流
        prefix: slider的前缀，用于区分不同的slider
    """
    pdf_bytes = file
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    
    if total_pages <= 1:
        page_number = 0
        if total_pages == 1:
            st.text(f"第1页 (共1页)")
    else:
        page_number = st.slider(
            f"选择页面 {prefix}",
            0,
            total_pages - 1,
            0,
            key=f"slider_{prefix}",
            label_visibility="collapsed"
        )
    
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    st.image(
        img,
        channels="RGB",
        use_container_width=True,
        output_format="PNG"
    )


def main():

    if 'mono_result' not in st.session_state:
        st.session_state.mono_result = None
    if 'dual_result' not in st.session_state:
        st.session_state.dual_result = None

    with st.sidebar:
        st.title("参数设置")

        uploaded_file = st.file_uploader(
            "上传PDF文档",
            type=['pdf'],
            help="请上传需要翻译的PDF文档"
        )

        service = st.selectbox(
            "翻译服务",
            options=[
                "google",
                "bing",
                "LLM",
            ],
            help="选择翻译服务提供商"
        )

        if service == "LLM":
            serveice_name = 'openai'
            st.subheader("LLM配置")
            base_url = st.text_input(
                "OPENAI_BASE_URL",
                value="https://api.zhizengzeng.com/v1",
                help="API基础URL"
            )
            api_key = st.text_input(
                "OPENAI_API_KEY",
                type="password",
                help="API密钥"
            )
            model = st.text_input(
                "OPENAI_MODEL",
                value="gpt-4o-mini",
                help="模型名称"
            )
        else:
            serveice_name = service

        st.subheader("语言设置")
        source_language = st.selectbox(
            "源语言",
            options=["英文", "中文", "日文", "韩文", "法文", "德文"],
            index=0
        )
        target_language = st.selectbox(
            "目标语言",
            options=["中文", "英文", "日文", "韩文", "法文", "德文"],
            index=0
        )

        st.subheader("高级选项")
        thread_count = st.number_input(
            "线程数",
            min_value=1,
            max_value=8,
            value=4,
            help="翻译时调用线程数"
        )

        if uploaded_file:
            translate_button = st.button("开始翻译", type="primary", use_container_width=True)
            cancel_button = st.button("取消", type="secondary", use_container_width=True)

    st.markdown("""
    <h1 style='text-align: center;'>
        PDF固版翻译
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **保留PDF格式的翻译器** 🌟
        
        🎯 **源项目地址**：https://github.com/Byaidu/PDFMathTranslate

        📌 **操作指南**：
        1. 上传需要翻译的PDF文档
        2. 选择源语言与目标语言
        3. 根据需求选择翻译服务商
        4. 实时预览对比翻译效果
        5. 下载单语/双语版本

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 20px;">
            📑 典型应用场景：<br>
            • 技术文档多语言本地化<br>
            • 学术论文格式保持翻译<br>
            • 商业合同精准术语转换<br>
            每次翻译都是专业级输出！
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 原文")
            display_pdf(uploaded_file.getvalue(), prefix="original")

        with col2:
            st.markdown("### 译文")
            if st.session_state.mono_result is not None:
                display_pdf(st.session_state.mono_result, prefix="translated")

                st.success("翻译完成！请选择下载版本：")
                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        label="下载单语版本",
                        data=st.session_state.mono_result,
                        file_name=f"translated_mono_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with download_col2:
                    st.download_button(
                        label="下载双语版本",
                        data=st.session_state.dual_result,
                        file_name=f"translated_dual_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("请在左侧设置参数并点击翻译按钮开始翻译")

        lang_code_map = {
            "中文": "zh", "英文": "en", "日文": "ja",
            "韩文": "ko", "法文": "fr", "德文": "de"
        }

        if translate_button:
            with st.spinner('正在翻译中,请耐心等待...'):
                lang_in = lang_code_map[source_language]
                lang_out = lang_code_map[target_language]

                mono_result, dual_result = translate_pdf(
                    file=uploaded_file,
                    lang_in=lang_in,
                    lang_out=lang_out,
                    service=serveice_name,
                    thread=thread_count,
                    api_key=api_key if 'api_key' in locals() else None,
                    base_url=base_url if 'base_url' in locals() else None,
                    model=model if 'model' in locals() else None
                )
                st.session_state.mono_result = mono_result
                st.session_state.dual_result = dual_result
                st.rerun()
    else:
        st.info("👈 请在左侧上传PDF文件并设置翻译参数")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'PDFTranslator'
current_page = 'PDFTranslator'
if current_page != st.session_state.previous_page:
        st.session_state.clear()
        st.session_state.previous_page = current_page
main()
