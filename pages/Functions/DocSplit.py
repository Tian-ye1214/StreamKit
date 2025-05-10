import re


async def split_tex_into_paragraphs(tex_content):
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
        if len(p) < 150 and not re.search(r'[a-zA-Z\u4e00-\u9fa5]', p):
            continue
        filtered_paragraphs.append(p)

    # 如果没有找到任何段落，返回整个内容作为一个段落
    if not filtered_paragraphs:
        filtered_paragraphs = [tex_content]

    return filtered_paragraphs


async def split_doc_into_paragraphs(doc_content):
    paragraphs = [p.strip() for p in doc_content.split('\n\n\n') if p.strip()]

    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in doc_content.split('\n\n') if p.strip()]

    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in doc_content.split('\n') if p.strip()]

    filtered_paragraphs = [p for p in paragraphs if p and len(p) > 100]
    if not filtered_paragraphs:
        filtered_paragraphs = [doc_content]
    
    return filtered_paragraphs
