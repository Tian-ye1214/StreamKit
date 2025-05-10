import re


async def split_tex_into_paragraphs(tex_content):
    # 删除所有注释行
    tex_content = re.sub(r'^%.*$', '', tex_content, flags=re.MULTILINE)
    
    # 删除所有以\开头的行
    tex_content = re.sub(r'^\\.*$', '', tex_content, flags=re.MULTILINE)
    
    # 删除空行
    tex_content = re.sub(r'\n\s*\n', '\n', tex_content)
    
    common_commands = []
    
    # 直接应用所有替换
    for command in common_commands:
        tex_content = re.sub(command, '', tex_content)

    paragraphs = [p.strip() for p in tex_content.split('\n\n') if p.strip()]

    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in tex_content.split('\n') if p.strip()]

    # 过滤掉只包含命令的段落（如只有\section{}的段落）
    filtered_paragraphs = []
    for p in paragraphs:
        # 如果段落不以英文字母开头，则跳过
        if not p.strip()[0].isalpha():
            continue
        # 如果段落只包含单一的命令（如\section{}），或者是空段落，则跳过
        if not p or re.match(r'^\s*\\(section|subsection|subsubsection|paragraph|title|author|date|label)\{.*?\}\s*$', p):
            continue
        # 如果段落长度过短且不包含实质性文本内容，则跳过
        if len(p) < 100:
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
