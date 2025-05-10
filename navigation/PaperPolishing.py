import streamlit as st
import os
from openai import AsyncOpenAI
import language_tool_python
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING
from pages.Functions.Prompt import polishing_prompt, political_prompt, grammer_prompt
from pages.Functions.DocSplit import split_tex_into_paragraphs, split_doc_into_paragraphs
import json
import io
from docx import Document
import asyncio

st.set_page_config(
    page_title="语法检查与文段润色",
    layout="wide",
    initial_sidebar_state="expanded"
)


async def initialization():
    if "Client" not in st.session_state:
        st.session_state.Client = AsyncOpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))
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
    if "file_type" not in st.session_state:
        st.session_state.file_type = None
    if "polished_paragraph_indices" not in st.session_state:
         st.session_state.polished_paragraph_indices = set()

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


async def polish_text_with_llm(message, temperature=0.6):
    try:
        st.subheader("润色结果")
        content = ""
        reasoning_content = ""
        model = st.session_state.get("selected_model", "deepseek-chat")
        with st.container(height=300):
            reason_placeholder = st.empty()
            message_placeholder = st.empty()
            async for chunk in await st.session_state.Client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
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


async def process_uploaded_file():
    """处理上传的文件，根据文件类型提取内容"""
    if st.session_state.uploaded_file is None:
        return None

    file_name = st.session_state.uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension == '.tex':
        st.session_state.file_type = 'tex'
        return st.session_state.uploaded_file.getvalue().decode('utf-8')
    elif file_extension == '.doc' or file_extension == '.docx':
        st.session_state.file_type = 'doc'
        try:
            doc = Document(io.BytesIO(st.session_state.uploaded_file.getvalue()))
            full_text = '\n\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return full_text
        except Exception as e:
            st.error(f"处理Word文档时出错: {str(e)}")
            return None
    else:
        st.error(f"不支持的文件类型: {file_extension}")
        return None


async def TEX_Polishing():
    col1, col2 = st.columns([1, 1])
    # --- 左栏：展示段落列表 ---
    with col1:
        if st.session_state.TEX_content is None:
            st.session_state.TEX_content = await process_uploaded_file()

        if st.session_state.TEX_content is not None:
            # 根据文件类型选择不同的段落分割方法
            if st.session_state.file_type == 'tex':
                st.session_state.tex_paragraphs = await split_tex_into_paragraphs(st.session_state.TEX_content)
                st.subheader("TEX 文件内容（原始段落）")
            else:
                st.session_state.tex_paragraphs = await split_doc_into_paragraphs(st.session_state.TEX_content)
                st.subheader("Word 文档内容（原始段落）")

        with st.container(height=800, border=False):
            # 如果当前正在润色某个段落，只显示该段落
            if st.session_state.current_polishing_paragraph_index is not None and not st.session_state.show_all_paragraphs:
                i = st.session_state.current_polishing_paragraph_index
                paragraph = st.session_state.tex_paragraphs[i]
                with st.container(border=True, height=600):
                    bg_color = "#e6ffe6" if i in st.session_state.polished_paragraph_indices else "white"
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <strong>段落 {i + 1}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("↩️ 返回所有段落", key="back_to_all_paragraphs"):
                        st.session_state.show_all_paragraphs = True
                        st.rerun()
                    modified_paragraph = st.text_area("段落详细内容", value=paragraph, height=500,
                                                      key=f"Paragraph_details_{i}", disabled=False)
                    st.session_state.paragraph_to_polish = modified_paragraph
                    st.session_state.tex_paragraphs[i] = modified_paragraph
                    st.session_state.TEX_content = st.session_state.TEX_content.replace(paragraph, modified_paragraph,
                                                                                        1)
            else:
                for i, paragraph in enumerate(st.session_state.tex_paragraphs):
                    bg_color = "#e6ffe6" if i in st.session_state.polished_paragraph_indices else "white"
                    with st.container(border=True, height=300):
                        st.markdown(f"""
                         <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                             <strong>段落 {i + 1}</strong>
                         </div>
                         """, unsafe_allow_html=True)
                        modified_paragraph = st.text_area("", value=paragraph, height=250, key=f"paragraph_{i}",
                                                          disabled=False)
                        st.session_state.tex_paragraphs[i] = modified_paragraph
                        st.session_state.TEX_content = st.session_state.TEX_content.replace(paragraph,
                                                                                            modified_paragraph, 1)
                        if st.button(f"润色该段落", key=f"polish_btn_{i}"):
                            st.session_state.current_polishing_paragraph_index = i
                            st.session_state.paragraph_to_polish = modified_paragraph
                            st.session_state.show_all_paragraphs = False
                            st.rerun()

        if st.session_state.tex_history:
            if st.sidebar.button(f"↩️ 撤销上次覆盖 ({len(st.session_state.tex_history)}步可撤销)",
                                 key="undo_overwrite_button"):
                st.session_state.TEX_content = st.session_state.tex_history.pop()
                st.success("已撤销上次覆盖操作！")
                st.session_state.paragraph_to_polish = ""
                st.session_state.polished_paragraph = None
                st.session_state.tex_paragraphs = await split_tex_into_paragraphs(st.session_state.TEX_content)
                st.session_state.polished_paragraph_indices = set()
                st.rerun()
        else:
            st.sidebar.button("↩️ 撤销上次覆盖", key="undo_overwrite_button_disabled", disabled=True)

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

            if st.button("开始润色", key="start_polish_button"):
                st.session_state.polished_paragraph = None
                with st.spinner("正在调用 LLM 进行润色..."):
                    message = polishing_prompt(st.session_state.paragraph_to_polish, st.session_state.polishing_prompt)
                    st.session_state.polished_paragraph = await polish_text_with_llm(message, temperature=0.8)

            if st.session_state.polished_paragraph is not None:
                polish_col1, polish_col2 = st.columns(2)
                with polish_col1:
                    if st.button("✅ 应用润色结果", key="apply_polish_button"):
                        st.session_state.tex_history.append(st.session_state.TEX_content)

                        original_paragraph = st.session_state.tex_paragraphs[
                            st.session_state.current_polishing_paragraph_index]

                        try:
                            new_content = st.session_state.TEX_content.replace(
                                original_paragraph,
                                st.session_state.polished_paragraph,
                                1
                            )

                            if new_content == st.session_state.TEX_content:
                                st.error("替换失败：请尝试手动更新")
                            else:
                                st.session_state.TEX_content = new_content
                                st.session_state.tex_paragraphs[
                                    st.session_state.current_polishing_paragraph_index] = st.session_state.polished_paragraph
                                st.session_state.polished_paragraph_indices.add(
                                    st.session_state.current_polishing_paragraph_index)
                                st.success("段落已更新！")
                                st.session_state.paragraph_to_polish = ""
                                st.session_state.polished_paragraph = None
                                st.session_state.current_polishing_paragraph_index = None
                                st.session_state.show_all_paragraphs = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"应用修改时出错: {str(e)}")

                with polish_col2:
                    if st.button("❌ 取消", key="cancel_polish_button"):
                        st.session_state.polished_paragraph = None
                        st.session_state.paragraph_to_polish = ""
                        st.session_state.current_polishing_paragraph_index = None
                        st.session_state.show_all_paragraphs = True
                        st.rerun()

        elif st.session_state.TEX_content is not None:
            st.info("请从左侧选择一个段落进行润色")


async def Textual_polishing():
    st.subheader("文本拼写检查")
    input_text = st.text_area(
        "请输入需要检查的文本：",
        height=200,
        key="text_input"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        check_button = st.button("LanguageTool语法检查", key="check_button")
    with col2:
        ai_check_button = st.button("AI语法检查", key="AI_check_button")
    with col3:
        ai_polish_button = st.button("AI文段润色", key="AI_Polishing_button")

    if check_button:
        col1, col2 = st.columns([1, 1])
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(input_text)
            tool.close()

            # 过滤掉所有Possible spelling mistake found类型的错误
            filtered_matches = [match for match in matches if match.message != 'Possible spelling mistake found.']

            if not filtered_matches:
                st.success("未发现语法或拼写错误！")
            else:
                with col1:
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

    if ai_check_button:
        if not input_text:
            st.warning("请输入需要检查的文本！")
            return

        st.subheader("AI语法审查结果")
        with st.spinner("正在使用AI进行语法审查..."):
            try:
                message = grammer_prompt(input_text)
                result = await polish_text_with_llm(message, temperature=0.1)
                try:
                    result_json = json.loads(result)
                    st.markdown("### 修正详情")
                    for i, correction in enumerate(result_json.get("corrections", [])):
                        st.markdown(f"""
                         <div style="font-size: 14px; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 4px;">
                             <div><span style="font-weight: bold;">错误内容</span>: <span style="color: #ff4b4b;">{correction.get('original', '')}</span></div>
                             <div><span style="font-weight: bold;">修正建议</span>: <span style="color: #00aa00;">{correction.get('corrected', '')}</span></div>
                             <div><span style="font-weight: bold;">错误说明</span>: {correction.get('explanation', '')}</div>
                         </div>
                         """, unsafe_allow_html=True)

                    st.markdown("### 完整修正文本")
                    st.markdown(result_json.get("corrected_text", ""))

                except json.JSONDecodeError:
                    st.markdown("### AI审查结果")
                    st.markdown(result)

            except Exception as e:
                st.error(f"AI语法审查过程中出现错误: {str(e)}")

    if ai_polish_button:
        if not input_text:
            st.warning("请输入需要检查的文本！")
            return
        try:
            message = polishing_prompt(input_text, st.session_state.polishing_prompt)
            await polish_text_with_llm(message, temperature=0.8)
        except Exception as e:
            st.warning("调用大模型出错:", e)


async def Political_censorship():
    if st.session_state.TEX_content is None:
        st.session_state.TEX_content = await process_uploaded_file()

    if st.session_state.TEX_content is not None:
        # 根据文件类型选择不同的段落分割方法
        if st.session_state.file_type == 'tex':
            st.session_state.tex_paragraphs = await split_tex_into_paragraphs(st.session_state.TEX_content)
        else:
            st.session_state.tex_paragraphs = await split_doc_into_paragraphs(st.session_state.TEX_content)

    col1, col2 = st.columns([1, 1])

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

    with col2:
        if st.session_state.current_censorship_paragraph_index is not None and st.session_state.paragraph_to_censor:
            st.subheader("段落政治审查")
            st.markdown(f"**当前审查段落 {st.session_state.current_censorship_paragraph_index + 1}**")

            st.session_state.censorship_prompt = st.text_area(
                "审查要求 (Prompt):",
                value=st.session_state.censorship_prompt,
                height=100,
                key="censorship_prompt_input"
            )

            if st.button("开始政治审查", key="start_censorship_button"):
                with st.spinner("正在调用大模型进行政治审查..."):
                    try:
                        message = political_prompt(st.session_state.censorship_prompt,
                                                   st.session_state.paragraph_to_censor)
                        content = await polish_text_with_llm(message, temperature=0.1)
                        st.session_state.censorship_results[
                            st.session_state.current_censorship_paragraph_index] = content
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


async def main():
    await initialization()
    st.markdown("""
    <h1 style='text-align: center;'>
        学术炼金术 -- AI论文润色
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        🌟 **AI论文润色助手使用指南** 🌟
        
        🧩 **功能亮点**：
        ✅ 支持TEX和Word文档格式<br>
        ✅ 智能语法检查和拼写纠错<br>
        ✅ AI辅助文段润色优化<br>
        ✅ 政治敏感内容审查<br>
        ✅ 多模型选择支持<br>

        📝 **操作流程**：
        1. 在左侧边栏上传论文文件（支持.tex/.doc/.docx格式）
        2. 选择适合的AI模型
        3. 使用以下功能：
           - 文段语法检查：快速检查文本语法和拼写
           - TEX文段润色：对论文进行智能润色
           - 政治审查：检查内容合规性

        🔍 **润色功能说明**：
        - 系统会自动将文档分割成段落
        - 点击"润色该段落"按钮进行单段润色
        - 可以自定义润色要求（Prompt）
        - 润色结果可以预览、应用或取消
        - 已润色段落会以绿色背景标记

        ⚠️ **注意事项**：
        - 润色过程会保持原文核心含义不变
        - 不会修改引用、段落标记和格式内容
        - 建议逐段润色并仔细检查结果
        - 可以随时撤销修改或清除所有记录

        💡 **小技巧**：
        - 使用"清除所有记录"按钮重置所有状态
        - 润色前可以调整AI模型以获得不同效果
        - 润色完成后可以下载处理后的文件
        """, unsafe_allow_html=True)

    with st.sidebar:
        def on_file_change():
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
            st.session_state.file_type = None
            if 'previous_TEX_content' in st.session_state:
                del st.session_state['previous_TEX_content']

        if st.button("清除所有记录"):
            keys_to_keep = ['previous_page']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            await initialization()
            st.rerun()

        st.session_state.uploaded_file = st.file_uploader(
            "选择文件",
            type=['tex', 'doc', 'docx'],
            key="file_uploader",
            on_change=on_file_change
        )

        st.markdown("### 模型选择")
        if st.session_state.TEX_content is not None:
            file_extension = '.tex' if st.session_state.file_type == 'tex' else '.docx'
            st.download_button(
                label=f"下载 {file_extension.upper()} 文件",
                data=st.session_state.TEX_content,
                file_name=f"{os.path.splitext(st.session_state.uploaded_file.name)[0]}_processed{file_extension}",
                mime="text/plain",
                key="download_tex_button"
            )

        model_names = list(HIGHSPEED_MODEL_MAPPING.keys())
        selected_model_name = st.selectbox(
            "选择模型",
            options=model_names,
            index=0
        )
        st.session_state.selected_model = HIGHSPEED_MODEL_MAPPING[selected_model_name]

    tab1, tab2, tab3 = st.tabs(['文段语法检查', 'tex文段润色', '政治审查'])
    with tab1:
        await Textual_polishing()

    with tab2:
        if st.session_state.uploaded_file is not None:
            await TEX_Polishing()
        else:
            st.warning("请先上传文件！")

    with tab3:
        if st.session_state.uploaded_file is not None:
            await Political_censorship()
        else:
            st.warning("请先上传文件！")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'PaperPolishing'
current_page = 'PaperPolishing'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
asyncio.run(main())
