import os
import streamlit as st
from PIL import Image
from pages.Functions.Prompt import Translator
import io
from openai import OpenAI
from pages.Functions.generation import diffusion_generation
from pages.Functions.controlnet_generation import style_transfer


def initialization():
    if "image" not in st.session_state:
        st.session_state.image = None
    if 'img_model' not in st.session_state:
        st.session_state.img_model = None
    if 'lora' not in st.session_state:
        st.session_state.lora = None
    if 'client' not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.environ.get("ZhiZz_API_KEY"), base_url=os.environ.get("ZhiZz_URL"))
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


def main():
    initialization()
    st.set_page_config(layout="wide", page_title="古建筑图像生成", page_icon="️")
    st.markdown("""
    <style>
        .stApp {
            background: #ececec !important;
        }

        section[data-testid="stSidebar"] {
            background: #ececec !important;
        }

        .stAppHeader {
            background-color: #ececec !important;
            color: #333333 !important;
        }

        [data-testid="stBottomBlockContainer"] {
            background: #ececec !important;
        }
    </style>
    """, unsafe_allow_html=True)

    _, col = st.columns([5, 1])
    with col:
        st.image("static/emblem.png", width=200)

    st.markdown("""
    <h1 style='text-align: center;'>
        古建筑图像生成
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.image is not None:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.image(st.session_state.image)
        with st.sidebar:
            image_bytes = io.BytesIO()
            st.session_state.image.save(image_bytes, format='PNG')
            st.download_button(
                label="下载图片",
                data=image_bytes.getvalue(),
                file_name="image.png",
                mime="image/png"
            )

    with st.sidebar:
        st.subheader("选择模型")
        st.session_state.img_model = "SDXL"
        st.subheader("配置参数")
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input(
                label='图像宽度',
                min_value=0,
                max_value=2048,
                value=1536,
                step=1,
                help='图像宽度'
            )
            num_inference_steps = st.number_input(
                label='推理步数',
                min_value=0,
                max_value=50,
                value=30,
                step=10,
                help='推理步数。值越大去噪次数越多，生成速度越慢。'
            )
        with col2:
            height = st.number_input(
                label='图像高度',
                min_value=0,
                max_value=2048,
                value=1024,
                step=1,
                help='图像高度'
            )
            guidance_scale = st.slider("文本指导强度", 1.0, 10.0, 7.5, help="文本指导强度。值越大文本约束越强，多样性越低。")

        st.subheader("选择风格lora")
        genres = st.multiselect(
            "选择想要组合的风格类别（最多选 2 个 LoRA）",
            ["无", "建筑风格", "绘画风格"],
        )

        selected_lora = []

        if "建筑风格" in genres:
            selected = st.multiselect(
                "建筑风格",
                options=["古建筑-正面", "3D渲染", "现代建筑渲染", "轴测图Axonometric", "概念草图", "未来风格",
                         "插画风格", "日本动漫风格"],
                default=["古建筑-正面"],
                max_selections=2
            )
            selected_lora.extend(selected)

        if "绘画风格" in genres:
            selected = st.multiselect(
                "绘画风格",
                options=["华岩", "梵高", "山水画"],
                default=[],
                max_selections=2
            )
            selected_lora.extend(selected)

        if len(selected_lora) > 2:
            st.error("最多只能选择 2 个 LoRA，请减少选择！")
            st.session_state.lora = selected_lora[:2]
            st.write("只保留:", selected_lora[:2])
        else:
            st.session_state.lora = selected_lora

        with st.expander("图片上传", expanded=False):
            st.session_state.uploaded_image = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )
            if st.session_state.uploaded_image:
                image = Image.open(st.session_state.uploaded_image)
                width, height = image.size
                if width > 256 or height > 256:
                    scale = 256 / max(height, width)
                    new_h, new_w = int(height * scale), int(width * scale)
                    image = image.resize((new_w, new_h), Image.BILINEAR)
                st.image(image, caption="图片预览")

    with st.expander("提示词参考"):
        st.markdown("""
***

# 古建筑风格提示词 (Prompts for Ancient Architecture Styles)

这是一份关于生成不同风格中国古建筑图像的提示词（Prompts）集合。

---

## 1. 写实风格古建筑

### **照片质感**

> 一张唐代寺庙建筑正面图，远景图，如山西南禅寺大殿，古朴木结构，高质量，真实性，高清。

> 正面视角，一座古老的唐代佛教寺庙，参考山西省南禅寺大殿，传统中国木结构建筑，屋顶有精美的斗拱和瓦当，背景是蓝天白云，远景是山脉，氛围宁静庄严，超高细节，超写实，8K高清，照片质感。

---

## 2. 特定设计风格

### **概念草图**

> 素描草图建筑风格，正面图，一座中国古代建筑。

> 中国山西古代建筑，传统木结构，带斗拱和飞檐，建筑概念草图风格，正面视角，黑白素描线稿，建筑设计手稿感，粗糙阴影线，富有细节，高清手绘草图。

### **动漫风格**

> 日本动漫建筑风格，正面图，明亮的色系，一座中国古代建筑。

> 日本动漫风格，正面图，色彩鲜艳，中国古建筑的绘画。

> 日本动漫建筑风格，山西古代建筑，传统中国木结构，飞檐斗拱，参考唐代寺庙风格，正面视角，色彩明亮，清新鲜艳的色调，蓝天白云背景，柔和光影，动漫插画风，细节丰富。

### **3D渲染风格**

> 3D渲染风格，中国古建筑。

---

## 3. 绘画艺术风格

### **中国山水画风格**

> 山西省一座重要的中国古建筑，中国佛教寺庙，中国古代山水画风格，高清，4k。

> 山西省佛教寺庙，中国古代山水画风格，水墨，松树，远山，古建筑，高清，4K。

### **梵高风格**

> 山西省一座重要的中国古建筑，中国佛教寺庙，梵高风格，高清，4k。

        """)

    negative_prompt = ("(worst quality, low quality:1.4), (bad anatomy), text, error, missing fingers, extra digit, "
                       "fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, deformed face")

    if user_input := st.chat_input("在这里输入您的提示词："):
        messages = Translator(user_input)
        model_parameter = {
            "model": 'kimi-k2-0711-preview',
            "messages": messages,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 4096
        }
        response = st.session_state.client.chat.completions.create(
            stream=False,
            **model_parameter
        )
        prompt = response.choices[0].message.content
        if prompt and st.session_state.uploaded_image is not None:
            save_dir = "/home/li/下载/uploads"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, st.session_state.uploaded_image.name)

            with open(save_path, "wb") as f:
                f.write(st.session_state.uploaded_image.getbuffer())
            with st.spinner('正在生成中,请耐心等待...'):
                st.session_state.image = style_transfer(
                    image=save_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    batch_size=1,
                    lora=st.session_state.lora
                )

        elif prompt and st.session_state.uploaded_image is None:
            with st.spinner('正在生成中,请耐心等待...'):
                if st.session_state.img_model == "SDXL":
                    st.session_state.image = diffusion_generation(

                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        batch_size=1,
                        lora=st.session_state.lora
                    )
                st.rerun()
        else:
            st.warning("请先填入提示词")


main()
