import re


async def split_tex_into_paragraphs(tex_content):
    tex_content = re.sub(r'^%.*$', '', tex_content, flags=re.MULTILINE)
    tex_content = re.sub(r'^\\.*$', '', tex_content, flags=re.MULTILINE)
    tex_content = re.sub(r'\n\s*\n', '\n', tex_content)

    paragraphs = [p.strip() for p in tex_content.split('\n\n') if p.strip()]

    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in tex_content.split('\n') if p.strip()]

    filtered_paragraphs = []
    for p in paragraphs:
        if not p.strip()[0].isalpha():
            continue
        if not p or re.match(r'^\s*\\(section|subsection|subsubsection|paragraph|title|author|date|label)\{.*?\}\s*$', p):
            continue
        if len(p) < 100:
            continue
        filtered_paragraphs.append(p)

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
