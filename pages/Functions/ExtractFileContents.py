import fitz
import pandas as pd
import docx
import base64
import re
import streamlit as st


@st.cache_data()
def encode_image_to_base64(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if isinstance(uploaded_file, str):
            with open(uploaded_file, "rb") as image_file:
                bytes_data = image_file.read()
        else:
            bytes_data = uploaded_file.getvalue()

        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"图片处理错误: {str(e)}")
        return None


def extract_text_from_pdf(file):
    try:
        content = ""
        pdf_bytes = file.getvalue()

        # 使用内存流打开PDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text = page.get_text()
                if text:
                    content += text + "\n"

        if not content.strip():
            raise ValueError("无法从PDF中提取文本内容")
        return content
    except Exception as e:
        print(f"PDF处理错误：{str(e)}")
        return None


def extract_text_from_excel(file):
    df = pd.read_excel(file)
    content = ""
    for column in df.columns:
        content += f"{column}:\n"
        content += df[column].to_string() + "\n\n"
    return content


def extract_text_from_docx(docx_file):
    """从Word文件中提取文本"""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(file):
    return file.getvalue().decode('utf-8')


async def extract_text(file):
    file_type = file.name.split('.')[-1].lower()
    try:
        if file_type == 'pdf':
            content = extract_text_from_pdf(file)
        elif file_type in ['doc', 'docx']:
            content = extract_text_from_docx(file)
        elif file_type in ['xlsx', 'xls']:
            content = extract_text_from_excel(file)
        elif file_type == 'txt':
            content = extract_text_from_txt(file)
        else:
            content = None
            raise ValueError(f"不支持的文件格式：{file_type}")
        content = re.sub(r'[ \t]{2,}', ' ', content)
        content = re.sub(r'^[ \t]+|[ \t]+$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n{2,}', '\n', content).strip('\n')
        content = re.sub(r'([,?!;:。.])\1+', r'\1', content)
        return content
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None


