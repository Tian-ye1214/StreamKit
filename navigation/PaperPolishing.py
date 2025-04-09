import streamlit as st
import os
import re
from openai import OpenAI
import language_tool_python

st.set_page_config(
    page_title="论文润色与处理工具",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialization():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "TEX_content" not in st.session_state:
        st.session_state.TEX_content = None
    if "tex_history" not in st.session_state:
        st.session_state.tex_history = []
    if "tex_paragraphs" not in st.session_state:
        st.session_state.tex_paragraphs = []
    if "current_polishing_paragraph_index" not in st.session_state:
        st.session_state.current_polishing_paragraph_index = None

    if "paragraph_to_polish" not in st.session_state:
        st.session_state.paragraph_to_polish = ""
    if "polishing_prompt" not in st.session_state:
        st.session_state.polishing_prompt = ("你是一位学术论文评审专家，你的任务评审下述文章段落，并："
                                             "1. 修正语法错误和拼写错误； "
                                             "2. 优化句式结构提升流畅性； "
                                             "3. 确保专业术语准确； "
                                             "4. 维持学术写作规范。请严格确保输出仅包含润色后的文段，不要包含任何解释、说明、注释等内容;"
                                             "5. 不要修改任何引用、段落标记内容，如\\cite、\\section。"
                                             "修改过程需保持原文核心含义不变，不得更改专业术语和数据信息。")
    if "polished_paragraph" not in st.session_state:
        st.session_state.polished_paragraph = None


def polish_text_with_llm(text, prompt, model="Pro/deepseek-ai/DeepSeek-V3"):
    system_prompt = """
    You are a professional academic writing optimization assistant specializing in refining and proofreading academic document excerpts. 
    Your core tasks are: 
    1. Correct grammatical errors and spelling mistakes 
    2. Optimize sentence structures for improved fluency 
    3. Ensure accurate usage of technical terminology 
    4. Maintain academic writing standards. Strictly provide ONLY the polished text without any explanations, notes, or additional comments. 
    Preserve the original meaning and never alter technical terms or numerical data. 
    When encountering ambiguous content requiring author confirmation, directly implement the most reasonable revision.
    """
    try:
        client = OpenAI(api_key=os.environ.get('SiliconFlow_API_KEY'), base_url=os.environ.get('SiliconFlow_URL'))
        st.subheader("润色结果")
        content = ""
        reasoning_content = ""
        with st.container(height=300):
            reason_placeholder = st.empty()
            message_placeholder = st.empty()
            for chunk in client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{prompt}\n\n文段内容：{text}"}
                    ],
                    temperature=1.0,
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

        return content
    except Exception as e:
        st.error(f"调用大模型时出错: {str(e)}")
        return None


def split_tex_into_paragraphs(tex_content):
    # 移除常见的命令和环境声明
    tex_content = re.sub(r'(?<!\\)%.*$', '', tex_content, flags=re.MULTILINE)
    tex_content = re.sub(r'\\documentclass.*?{.*?}', '', tex_content)
    tex_content = re.sub(r'\\usepackage(\[.*?\])?{.*?}', '', tex_content)
    tex_content = re.sub(r'\\begin{document}', '', tex_content)
    tex_content = re.sub(r'\\end{document}', '', tex_content)
    tex_content = re.sub(r'\\maketitle', '', tex_content)
    tex_content = re.sub(r'\\usetikzlibrary.*?$', '', tex_content, flags=re.MULTILINE)
    tex_content = re.sub(r'\\bibliographystyle\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\bibliography\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\tableofcontents', '', tex_content)
    tex_content = re.sub(r'\\listoffigures', '', tex_content)
    tex_content = re.sub(r'\\listoftables', '', tex_content)
    tex_content = re.sub(r'\\setcounter\{.*?\}\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\graphicspath\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\newcommand\{.*?\}\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\renewcommand\{.*?\}\{.*?\}', '', tex_content)
    tex_content = re.sub(r'\\label\{.*?\}', '', tex_content)

    paragraphs = [p.strip() for p in tex_content.split('\n\n') if p.strip()]
    
    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in tex_content.split('\n') if p.strip()]

    # 过滤掉只包含命令的段落（如只有\section{}的段落）
    filtered_paragraphs = []
    for p in paragraphs:
        # 如果段落只包含单一的命令（如\section{}），或者是空段落，则跳过
        if not p or re.match(r'^\s*\\(section|subsection|subsubsection|paragraph|title|author|date|label)\{.*?\}\s*$',
                             p):
            continue
        # 如果段落长度过短且不包含实质性文本内容，则跳过
        if len(p) < 10 and not re.search(r'[a-zA-Z\u4e00-\u9fa5]', p):
            continue
        filtered_paragraphs.append(p)
    
    # 如果没有找到任何段落，返回整个内容作为一个段落
    if not filtered_paragraphs:
        filtered_paragraphs = [tex_content]
        
    return filtered_paragraphs


def TEX_Polishing():
    col1, col2 = st.columns([3, 2])
    # --- 左栏：展示段落列表 ---
    with col1:
        if st.session_state.TEX_content is not None:
            st.subheader("TEX 文件内容（原始段落）")

            if not st.session_state.tex_paragraphs:
                st.session_state.tex_paragraphs = split_tex_into_paragraphs(st.session_state.TEX_content)

            with st.container(height=800, border=False):
                for i, paragraph in enumerate(st.session_state.tex_paragraphs):
                    with st.container(border=True):
                        # 使用markdown显示段落，保留格式
                        st.markdown(f"**段落 {i+1}**")
                        st.text_area("", value=paragraph, height=100, key=f"paragraph_{i}", disabled=True)
                        if st.button(f"润色该段落", key=f"polish_btn_{i}"):
                            st.session_state.current_polishing_paragraph_index = i
                            st.session_state.paragraph_to_polish = paragraph
                            st.rerun()

            st.sidebar.download_button(
                label="下载 TEX 文件",
                data=st.session_state.TEX_content,
                file_name=f"{os.path.splitext(st.session_state.uploaded_file.name)[0]}_processed.tex",
                mime="text/plain",
                key="download_tex_button"
            )

            if st.session_state.tex_history:
                if st.sidebar.button(f"↩️ 撤销上次覆盖 ({len(st.session_state.tex_history)}步可撤销)",
                                     key="undo_overwrite_button"):
                    st.session_state.TEX_content = st.session_state.tex_history.pop()
                    st.success("已撤销上次覆盖操作！")
                    st.session_state.paragraph_to_polish = ""
                    st.session_state.polished_paragraph = None
                    st.session_state.tex_paragraphs = split_tex_into_paragraphs(st.session_state.TEX_content)
                    st.rerun()
            else:
                st.sidebar.button("↩️ 撤销上次覆盖", key="undo_overwrite_button_disabled", disabled=True)

        elif st.session_state.uploaded_file is not None:
            st.session_state.TEX_content = st.session_state.uploaded_file.getvalue().decode('utf-8')
            st.session_state.tex_history = []
            st.session_state.tex_paragraphs = []
            st.rerun()

    # --- 右栏：润色交互 ---
    with col2:
        if st.session_state.current_polishing_paragraph_index is not None and st.session_state.paragraph_to_polish:
            st.subheader("段落润色")

            st.session_state.polishing_prompt = st.text_area(
                "润色要求 (Prompt):",
                value=st.session_state.polishing_prompt,
                height=100,
                key="prompt_input"
            )

            # 润色按钮
            if st.button("开始润色", key="start_polish_button"):
                st.session_state.polished_paragraph = None
                with st.spinner("正在调用 LLM 进行润色..."):
                    st.session_state.polished_paragraph = polish_text_with_llm(
                        st.session_state.paragraph_to_polish,
                        st.session_state.polishing_prompt
                    )

            if st.session_state.polished_paragraph is not None:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 应用润色结果", key="apply_polish_button"):
                        st.session_state.tex_history.append(st.session_state.TEX_content)

                        original_paragraph = st.session_state.tex_paragraphs[
                            st.session_state.current_polishing_paragraph_index]
                        
                        # 使用更精确的替换方法
                        try:
                            # 确保段落内容被正确替换
                            new_content = st.session_state.TEX_content.replace(
                                original_paragraph,
                                st.session_state.polished_paragraph,
                                1
                            )
                            
                            # 检查替换是否成功
                            if new_content == st.session_state.TEX_content:
                                st.error("替换失败：未找到匹配的段落。请尝试重新选择段落。")
                            else:
                                st.session_state.TEX_content = new_content
                                st.session_state.tex_paragraphs[
                                    st.session_state.current_polishing_paragraph_index] = st.session_state.polished_paragraph
                                st.success("段落已更新！")
                                st.session_state.paragraph_to_polish = ""
                                st.session_state.polished_paragraph = None
                                st.session_state.current_polishing_paragraph_index = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"应用修改时出错: {str(e)}")

                with col2:
                    if st.button("❌ 取消", key="cancel_polish_button"):
                        st.session_state.polished_paragraph = None
                        st.session_state.paragraph_to_polish = ""
                        st.session_state.current_polishing_paragraph_index = None
                        st.rerun()

        elif st.session_state.TEX_content is not None:
            st.info("请从左侧选择一个段落进行润色")


def Textual_polishing():
    st.subheader("文本拼写检查")
    input_text = st.text_area(
        "请输入需要检查的文本：",
        height=200,
        key="text_input"
    )
    if st.button("开始检查", key="check_button"):
        col1, col2 = st.columns([1, 1])
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(input_text)
            tool.close()

            if matches:
                with col1:
                    st.markdown("### 错误详情")
                    for i, match in enumerate(matches):
                        if match.message == 'Possible spelling mistake found.':
                            continue
                        st.markdown(f"""
                        <div style="font-size: 14px; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 4px;">
                            <div><span style="font-weight: bold;">错误类型</span>: {match.ruleId}</div>
                            <div><span style="font-weight: bold;">错误位置</span>: {match.offset}-{match.offset + match.errorLength}</div>
                            <div><span style="font-weight: bold;">错误内容</span>: <span style="color: #ff4b4b;">{input_text[match.offset:match.offset + match.errorLength]}</span></div>
                            <div><span style="font-weight: bold;">建议修改</span>: {match.replacements[0] if match.replacements else '无建议'}</div>
                            <div><span style="font-weight: bold;">错误说明</span>: {match.message}</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col2:
                    st.markdown("### 逐条修改预览")
                    for i, match in enumerate(matches):
                        if match.message == 'Possible spelling mistake found.':
                            continue
                        if match.replacements:
                            modified_segment = (
                                    input_text[:match.offset] +
                                    f"**{match.replacements[0]}**" +  # 加粗修改部分
                                    input_text[match.offset + match.errorLength:]
                            )
                            st.markdown(modified_segment)
                            st.markdown('--------------------')

                st.markdown("### 最终修改结果")
                corrected_text = input_text
                for match in sorted(matches, key=lambda x: -x.offset):
                    if match.message == 'Possible spelling mistake found.':
                        continue
                    if match.replacements:
                        corrected_text = (
                                corrected_text[:match.offset] +
                                f"**{match.replacements[0]}**" +
                                corrected_text[match.offset + match.errorLength:]
                        )
                st.markdown(corrected_text)

            else:
                st.success("未发现语法或拼写错误！")

        except Exception as e:
            st.error(f"检查过程中出现错误: {str(e)}")


def main():
    initialization()
    st.title("论文润色与处理工具")

    if st.sidebar.button("清除所有记录"):
        keys_to_keep = ['previous_page']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        initialization()
        st.rerun()

    with st.sidebar:
        def on_file_change():
            st.session_state.TEX_content = None
            st.session_state.paragraph_to_polish = ""
            st.session_state.polished_paragraph = None
            st.session_state.tex_history = []
            st.session_state.tex_paragraphs = []
            st.session_state.current_polishing_paragraph_index = None
            if 'previous_TEX_content' in st.session_state:
                del st.session_state['previous_TEX_content']

        st.session_state.uploaded_file = st.file_uploader(
            "选择文件",
            type=['tex'],
            key="file_uploader",
            on_change=on_file_change
        )

    tab1, tab2 = st.tabs(['文段语法检查', 'tex文段润色'])
    with tab1:
        Textual_polishing()

    with tab2:
        if st.session_state.uploaded_file is not None:
            if st.session_state.uploaded_file.name.endswith('.tex'):
                TEX_Polishing()


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'PaperPolishing'
current_page = 'PaperPolishing'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    initialization()
    st.session_state.previous_page = current_page
main()
