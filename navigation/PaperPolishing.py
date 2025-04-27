import streamlit as st
import os
import re
from openai import OpenAI
import language_tool_python
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING


def initialization():
    """
    初始化函数：设置应用程序的初始状态
    使用Streamlit的session_state来存储应用程序的状态信息
    保存内容为操作记录、状态信息、调用的LLM等
    """
    if "Client" not in st.session_state:
        st.session_state.Client = OpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))
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
    if "show_all_paragraphs" not in st.session_state:
        st.session_state.show_all_paragraphs = True

    if "paragraph_to_polish" not in st.session_state:
        st.session_state.paragraph_to_polish = ""
    if "polishing_prompt" not in st.session_state:
        st.session_state.polishing_prompt = ("你是一位学术论文评审专家，你的任务评审下述文章段落，并："
                                             "1. 修正语法错误和拼写错误； "
                                             "2. 优化句式结构提升流畅性； "
                                             "3. 确保专业术语准确； "
                                             "4. 维持学术写作规范。请严格确保输出仅包含润色后的文段，不要包含任何解释、说明、注释等内容;"
                                             "5. 不要修改任何引用、段落标记内容，如\\cite、\\section。"
                                             "6. 不要修改格式内容，如换行、缩进、空格。格式请与原文保持一致"
                                             "修改过程需保持原文核心含义不变，不得更改专业术语和数据信息。")
    if "polished_paragraph" not in st.session_state:
        st.session_state.polished_paragraph = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "deepseek-chat"
    if "censorship_prompt" not in st.session_state:
        st.session_state.censorship_prompt = ("你是一位政治审查专家，你的任务是对下述文章进行政治审查，并："
                                             "1. 识别可能存在的政治敏感内容；"
                                             "2. 评估内容的政治倾向性；"
                                             "3. 指出可能违反国家法律法规的内容；"
                                             "4. 提供详细的审查意见。")
    if "censorship_result" not in st.session_state:
        st.session_state.censorship_result = None
    if "current_censorship_paragraph_index" not in st.session_state:
        st.session_state.current_censorship_paragraph_index = None
    if "show_all_censorship_paragraphs" not in st.session_state:
        st.session_state.show_all_censorship_paragraphs = True
    if "paragraph_to_censor" not in st.session_state:
        st.session_state.paragraph_to_censor = ""
    if "censorship_results" not in st.session_state:
        st.session_state.censorship_results = {}


def polish_text_with_llm(text, prompt):
    """
    使用大语言模型对文本进行润色
    
    参数:
        text (str): 需要润色的文本内容
        prompt (str): 润色的提示词
    
    返回:
        str: 润色后的文本内容
    """
    # 系统提示词，指导大语言模型的行为
    system_prompt = """
    You are a professional academic writing optimization assistant specializing in refining and proofreading academic document excerpts. 
    Your core tasks are: 
    1. Correct grammatical errors and spelling mistakes 
    2. Optimize sentence structures for improved fluency 
    3. Ensure accurate usage of technical terminology 
    4. Maintain academic writing standards. Strictly provide ONLY the polished text without any explanations, notes, or additional comments. 
    5. Do not modify any citation or paragraph mark content, such as \\cite, \\section.
    6. Do not modify the formatting, such as line breaks, indents, and spaces. Please keep the format consistent with the original text.
    Preserve the original meaning and never alter technical terms or numerical data. 
    When encountering ambiguous content requiring author confirmation, directly implement the most reasonable revision.
    """
    try:
        st.subheader("润色结果")
        content = ""  # 存储润色后的内容
        reasoning_content = ""  # 存储推理过程内容

        # 获取选择的模型
        model = st.session_state.get("selected_model", "deepseek-chat")
            
        # 创建一个容器来显示润色结果
        with st.container(height=300):
            reason_placeholder = st.empty()  # 创建推理内容的占位符
            message_placeholder = st.empty()  # 创建润色结果的占位符
            
            # 调用大语言模型API进行润色
            for chunk in st.session_state.Client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{prompt}\n\n文段内容：{text}"}
                    ],
                    temperature=1.0,
                    stream=True
            ):
                # 处理每个返回的文本块
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
    """
    将TEX文件内容分割成段落
    
    参数:
        tex_content (str): TEX文件内容
    
    返回:
        list: 段落列表
    """
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

    # 按空行分割段落
    paragraphs = [p.strip() for p in tex_content.split('\n\n') if p.strip()]

    # 如果段落数量太少，尝试按换行符分割
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
    """
    TEX文件润色功能
    提供TEX文件的段落润色功能，包括段落选择、润色和应用
    """
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    # --- 左栏：展示段落列表 ---
    with col1:
        st.session_state.TEX_content = st.session_state.uploaded_file.getvalue().decode('utf-8') if st.session_state.TEX_content is None else st.session_state.TEX_content
        st.session_state.tex_paragraphs = split_tex_into_paragraphs(st.session_state.TEX_content)
        st.subheader("TEX 文件内容（原始段落）")

        # 创建段落显示容器
        with st.container(height=800, border=False):
            # 如果当前正在润色某个段落，只显示该段落
            if st.session_state.current_polishing_paragraph_index is not None and not st.session_state.show_all_paragraphs:
                i = st.session_state.current_polishing_paragraph_index
                paragraph = st.session_state.tex_paragraphs[i]
                with st.container(border=True, height=600):
                    st.markdown(f"**段落 {i + 1}**")
                    if st.button("↩️ 返回所有段落", key="back_to_all_paragraphs"):
                        st.session_state.show_all_paragraphs = True
                        st.rerun()
                    # 段落编辑区域
                    modified_paragraph = st.text_area("段落详细内容", value=paragraph, height=500, key=f"Paragraph_details_{i}", disabled=False)
                    st.session_state.paragraph_to_polish = modified_paragraph
                    st.session_state.tex_paragraphs[i] = modified_paragraph
                    st.session_state.TEX_content = st.session_state.TEX_content.replace(paragraph, modified_paragraph, 1)
            else:
                # 显示所有段落
                for i, paragraph in enumerate(st.session_state.tex_paragraphs):
                    with st.container(border=True, height=300):
                        st.markdown(f"**段落 {i + 1}**")
                        # 段落编辑区域
                        modified_paragraph = st.text_area("段落内容", value=paragraph, height=250, key=f"paragraph_{i}", disabled=False)
                        st.session_state.tex_paragraphs[i] = modified_paragraph
                        st.session_state.TEX_content = st.session_state.TEX_content.replace(paragraph, modified_paragraph, 1)
                        # 点击润色后，重置状态
                        if st.button(f"润色该段落", key=f"polish_btn_{i}"):
                            st.session_state.current_polishing_paragraph_index = i
                            st.session_state.paragraph_to_polish = modified_paragraph
                            st.session_state.show_all_paragraphs = False
                            st.rerun()

        # 下载按钮
        st.sidebar.download_button(
            label="下载 TEX 文件",
            data=st.session_state.TEX_content,
            file_name=f"{os.path.splitext(st.session_state.uploaded_file.name)[0]}_processed.tex",
            mime="text/plain",
            key="download_tex_button"
        )

        # 撤销操作按钮
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

    # --- 右栏：润色交互 ---
    with col2:
        if st.session_state.current_polishing_paragraph_index is not None and st.session_state.paragraph_to_polish:
            st.subheader("段落润色")

            # 润色提示词编辑区域
            st.session_state.polishing_prompt = st.text_area(
                "润色要求 (Prompt):",
                value=st.session_state.polishing_prompt,
                height=100,
                key="prompt_input"
            )

            # 开始润色按钮
            if st.button("开始润色", key="start_polish_button"):
                st.session_state.polished_paragraph = None
                with st.spinner("正在调用 LLM 进行润色..."):
                    st.session_state.polished_paragraph = polish_text_with_llm(
                        st.session_state.paragraph_to_polish,
                        st.session_state.polishing_prompt
                    )

            if st.session_state.polished_paragraph is not None:
                polish_col1, polish_col2 = st.columns(2)
                with polish_col1:
                    # 应用润色结果按钮
                    if st.button("✅ 应用润色结果", key="apply_polish_button"):
                        # 保存当前内容到历史记录
                        st.session_state.tex_history.append(st.session_state.TEX_content)

                        # 获取原始段落
                        original_paragraph = st.session_state.tex_paragraphs[
                            st.session_state.current_polishing_paragraph_index]

                        try:
                            # 替换段落内容
                            new_content = st.session_state.TEX_content.replace(
                                original_paragraph,
                                st.session_state.polished_paragraph,
                                1
                            )

                            # 检查替换是否成功
                            if new_content == st.session_state.TEX_content:
                                st.error("替换失败：请尝试手动更新")
                            else:
                                st.session_state.TEX_content = new_content
                                st.session_state.tex_paragraphs[
                                    st.session_state.current_polishing_paragraph_index] = st.session_state.polished_paragraph
                                st.success("段落已更新！")
                                # 重置状态
                                st.session_state.paragraph_to_polish = ""
                                st.session_state.polished_paragraph = None
                                st.session_state.current_polishing_paragraph_index = None
                                st.session_state.show_all_paragraphs = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"应用修改时出错: {str(e)}")

                with polish_col2:
                    # 取消按钮
                    if st.button("❌ 取消", key="cancel_polish_button"):
                        # 重置状态
                        st.session_state.polished_paragraph = None
                        st.session_state.paragraph_to_polish = ""
                        st.session_state.current_polishing_paragraph_index = None
                        st.session_state.show_all_paragraphs = True
                        st.rerun()

        elif st.session_state.TEX_content is not None:
            st.info("请从左侧选择一个段落进行润色")


def Textual_polishing():
    """
    文本拼写检查功能
    提供普通文本的语法和拼写检查功能
    """
    st.subheader("文本拼写检查")
    # 文本输入区域
    input_text = st.text_area(
        "请输入需要检查的文本：",
        height=200,
        key="text_input"
    )
    if st.button("开始检查", key="check_button"):
        col1, col2 = st.columns([1, 1])
        try:
            tool = language_tool_python.LanguageTool('en-US')
            # 检查文本
            matches = tool.check(input_text)
            tool.close()

            # 过滤掉所有Possible spelling mistake found类型的错误
            filtered_matches = [match for match in matches if match.message != 'Possible spelling mistake found.']
            
            if not filtered_matches:
                st.success("未发现语法或拼写错误！")
            else:
                with col1:
                    # 展示错误详情
                    st.markdown("### 错误详情")
                    for i, match in enumerate(filtered_matches):
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
                    # 修改预览
                    st.markdown("### 逐条修改预览")
                    for i, match in enumerate(filtered_matches):
                        if match.replacements:
                            modified_segment = (
                                    input_text[:match.offset] +
                                    f"**{match.replacements[0]}**" +
                                    input_text[match.offset + match.errorLength:]
                            )
                            st.markdown(modified_segment)
                            st.markdown('--------------------')

                st.markdown("### 最终修改结果")
                corrected_text = input_text
                for match in sorted(filtered_matches, key=lambda x: -x.offset):
                    if match.replacements:
                        corrected_text = (
                                corrected_text[:match.offset] +
                                f"**{match.replacements[0]}**" +
                                corrected_text[match.offset + match.errorLength:]
                        )
                st.markdown(corrected_text)

        except Exception as e:
            st.error(f"检查过程中出现错误: {str(e)}")


def Political_censorship():
    """
    政治审查功能
    提供TEX文件的政治审查功能，包括段落选择、审查和结果展示
    """
    # 获取TEX文件内容
    st.session_state.TEX_content = st.session_state.uploaded_file.getvalue().decode(
        'utf-8') if st.session_state.TEX_content is None else st.session_state.TEX_content
    st.session_state.tex_paragraphs = split_tex_into_paragraphs(st.session_state.TEX_content)

    col1, col2 = st.columns([1, 1])

    # 左栏：显示段落列表
    with col1:
        st.subheader("TEX 文件内容（原始段落）")
        with st.container(height=800, border=False):
            if st.session_state.current_censorship_paragraph_index is not None and not st.session_state.show_all_censorship_paragraphs:
                i = st.session_state.current_censorship_paragraph_index
                paragraph = st.session_state.tex_paragraphs[i]
                with st.container(border=True, height=600):
                    st.markdown(f"**段落 {i + 1}**")
                    if st.button("↩️ 返回所有段落", key="back_to_all_censorship_paragraphs"):
                        st.session_state.show_all_censorship_paragraphs = True
                        st.rerun()
                    st.markdown(paragraph)
                    st.session_state.paragraph_to_censor = paragraph
            else:
                for i, paragraph in enumerate(st.session_state.tex_paragraphs):
                    with st.container(border=True, height=300):
                        st.markdown(f"**段落 {i + 1}**")
                        st.markdown(paragraph)
                        if st.button(f"审查该段落", key=f"censor_btn_{i}"):
                            st.session_state.current_censorship_paragraph_index = i
                            st.session_state.show_all_censorship_paragraphs = False
                            st.rerun()

    # 右栏：审查交互
    with col2:
        if st.session_state.current_censorship_paragraph_index is not None and st.session_state.paragraph_to_censor:
            st.subheader("段落政治审查")
            st.markdown(f"**当前审查段落 {st.session_state.current_censorship_paragraph_index + 1}**")
            st.markdown(st.session_state.paragraph_to_censor)

            st.session_state.censorship_prompt = st.text_area(
                "审查要求 (Prompt):",
                value=st.session_state.censorship_prompt,
                height=100,
                key="censorship_prompt_input"
            )

            if st.button("开始政治审查", key="start_censorship_button"):
                with st.spinner("正在调用大模型进行政治审查..."):
                    system_prompt = """
                    你是一位政治审查专家，你的任务是对下述文章段落进行政治审查，并：
                    1. 识别可能存在的政治敏感内容；
                    2. 评估内容的政治倾向性；
                    3. 指出可能违反国家法律法规的内容；
                    4. 提供详细的审查意见。
                    请提供全面的审查报告，包括发现的问题和建议。
                    """
                    
                    try:
                        model = st.session_state.get("selected_model", "deepseek-chat")
                        content = ""
                        # 创建容器显示审查结果
                        with st.container(height=300):
                            message_placeholder = st.empty()
                            # 调用大语言模型API进行审查
                            for chunk in st.session_state.Client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"{st.session_state.censorship_prompt}\n\n段落内容：{st.session_state.paragraph_to_censor}"}
                                    ],
                                    temperature=0.7,
                                    stream=True
                            ):
                                # 处理每个返回的文本块
                                if chunk.choices and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    if delta and delta.content is not None:
                                        content += delta.content
                                        message_placeholder.markdown(
                                            f"<div style='font-size:16px; margin-top:10px;'>{content}</div>",
                                            unsafe_allow_html=True
                                        )

                        # 保存审查结果
                        st.session_state.censorship_results[st.session_state.current_censorship_paragraph_index] = content
                        st.session_state.censorship_result = content
                        st.rerun()
                    except Exception as e:
                        st.error(f"调用大模型时出错: {str(e)}")

            if st.session_state.current_censorship_paragraph_index in st.session_state.censorship_results:
                st.markdown("### 审查结果")
                st.markdown(st.session_state.censorship_results[st.session_state.current_censorship_paragraph_index])

                if st.button("返回所有段落", key="return_to_all_paragraphs_censorship"):
                    st.session_state.show_all_censorship_paragraphs = True
                    st.rerun()

        elif st.session_state.censorship_results:
            st.subheader("所有段落审查结果摘要")
            
            for i, paragraph in enumerate(st.session_state.tex_paragraphs):
                with st.expander(f"段落 {i + 1} 审查结果", expanded=False):
                    if i in st.session_state.censorship_results:
                        st.markdown(st.session_state.censorship_results[i])
                    else:
                        st.info("该段落尚未审查")
        
        else:
            st.info("请从左侧选择一个段落进行政治审查")


def main():
    """
    主函数：应用程序的入口点
    初始化应用程序并设置界面
    """
    # 初始化应用程序
    initialization()
    st.title("学术炼金术")

    # 添加使用说明
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **论文润色与处理工具使用指南** 🌟
        
        🧩 **系统亮点**：
        
        ✅ 智能润色学术论文<br>
        ✅ 语法与拼写检查<br>
        ✅ 政治内容审查<br>
        ✅ 支持TEX格式文件处理<br>

        🛠️ **功能模块**：
        
        1️⃣ **文段语法检查**：<br>
        - 输入任意文本进行语法和拼写检查<br>
        - 显示详细错误信息及修改建议<br>
        - 提供逐条修改预览和最终修改结果<br>
        
        2️⃣ **TEX文段润色**：<br>
        - 上传TEX格式论文文件<br>
        - 智能分段处理论文内容<br>
        - 使用AI模型进行专业润色<br>
        - 支持逐段润色和整体应用<br>
        - 提供润色历史记录和撤销功能<br>
        
        3️⃣ **政治审查**：<br>
        - 对论文内容进行政治敏感性审查<br>
        - 识别潜在的政治敏感内容<br>
        - 提供详细的审查报告<br>

        📝 **操作流程**：
        1. 在侧边栏选择模型和上传TEX文件
        2. 选择需要使用的功能模块
        3. 按照界面提示进行操作
        4. 下载处理后的文件

        💡 **使用技巧**：
        - 润色前可自定义润色提示词<br>
        - 使用撤销功能可恢复之前的修改<br>
        - 可随时下载处理后的文件保存结果
        """, unsafe_allow_html=True)

    # 清除所有记录按钮
    if st.sidebar.button("清除所有记录"):
        keys_to_keep = ['previous_page']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        initialization()
        st.rerun()

    # 侧边栏设置
    with st.sidebar:
        # 文件上传回调函数
        def on_file_change():
            # 重置状态
            st.session_state.TEX_content = None
            st.session_state.paragraph_to_polish = ""
            st.session_state.polished_paragraph = None
            st.session_state.tex_history = []
            st.session_state.tex_paragraphs = []
            st.session_state.current_polishing_paragraph_index = None
            st.session_state.current_censorship_paragraph_index = None
            st.session_state.show_all_censorship_paragraphs = True
            st.session_state.paragraph_to_censor = ""
            st.session_state.censorship_results = {}
            st.session_state.censorship_result = None
            if 'previous_TEX_content' in st.session_state:
                del st.session_state['previous_TEX_content']

        # 文件上传器
        st.session_state.uploaded_file = st.file_uploader(
            "选择文件",
            type=['tex'],
            key="file_uploader",
            on_change=on_file_change
        )

        # 模型选择
        st.sidebar.markdown("### 模型选择")

        model_names = list(HIGHSPEED_MODEL_MAPPING.keys())
        selected_model_name = st.sidebar.selectbox(
            "选择模型",
            options=model_names,
            index=0
        )
        st.session_state.selected_model = HIGHSPEED_MODEL_MAPPING[selected_model_name]

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(['文段语法检查', 'tex文段润色', '政治审查'])
    with tab1:
        Textual_polishing()

    with tab2:
        if st.session_state.uploaded_file is not None:
            TEX_Polishing()
        else:
            st.warning("请先上传文件！")

    with tab3:
        if st.session_state.uploaded_file is not None:
            Political_censorship()
        else:
            st.warning("请先上传文件！")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'PaperPolishing'
current_page = 'PaperPolishing'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page

main()
