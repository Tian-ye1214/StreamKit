import os
import requests
import time
import streamlit as st
from urllib import request
from PIL import Image
from pages.Functions.Prompt import Translator
import io
from openai import OpenAI
from google import genai


def initialization():
    if "image" not in st.session_state:
        st.session_state.image = None
    if 'img_model' not in st.session_state:
        st.session_state.img_model = None
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=os.environ.get("ZhiZz_API_KEY"),
                                                    base_url=os.environ.get("ZhiZz_URL"))
    if 'google_client' not in st.session_state:
        st.session_state.google_client = genai.Client(
            api_key=os.environ.get("ZhiZz_API_KEY"),
            http_options={
                "base_url": "https://api.zhizengzeng.com/google"
            },
        )
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


def flux_generation(model_name, prompt, width, height):
    info_placeholder = st.empty()
    flux_url = os.path.join(os.environ["FLUX_URL"], model_name)

    gen_request = requests.post(
        flux_url,
        headers={
            'accept': 'application/json',
            'x-key': os.environ.get("FLUX_API_KEY"),
            'Content-Type': 'application/json',
        },
        json={
            'prompt': prompt,
            'width': width,
            'height': height,
        },
    ).json()
    request_id = gen_request["id"]
    st.markdown(gen_request)

    while True:
        time.sleep(2)
        result = requests.get(
            os.path.join(os.environ["FLUX_URL"], 'get_result'),
            headers={
                'accept': 'application/json',
                'x-key': os.environ.get("FLUX_API_KEY"),
            },
            params={
                'id': request_id,
            },
        ).json()
        if result["status"] == "Ready":
            info_placeholder.markdown(f"Result: {result['result']['sample']}")
            img_url = result['result']['sample']
            return img_url
        else:
            info_placeholder.markdown(f"Status: {result['status']}")


def main():
    initialization()
    st.set_page_config(layout="wide", page_title="AI图像生成", page_icon="️")

    st.markdown("""
    <h1 style='text-align: center;'>
        AI图像生成
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.image is not None:
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
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
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("static/Original.jpg", caption="原始图片", width=300)
        with col2:
            st.image("static/AIGeneration.jpg", caption="Nano Banana生成", width=300)

    with st.sidebar:
        st.subheader("选择模型")
        st.session_state.img_model = st.selectbox(
            "模型",
            options=["FLUX", "Nano Banana", "doubao-seedream-3.0-t2i", "gpt-image-1", "dall-e-3"],
            help="选择图像生成模型",
            index=1,
        )

        if st.session_state.img_model == "FLUX":
            st.subheader("选择类型")
            model_name = st.selectbox(
                "type",
                options=["flux-pro-1.1", "flux-pro", "flux-dev"],
                index=2
            )
        st.subheader("配置参数")
        resulution = st.selectbox(
            "分辨率",
            options=["256x256", "512x512", "1024x1024"],
            help="生成图像的分辨率",
            index=2
        )
        width, height = resulution.split('x')[0], resulution.split('x')[1]

        if st.session_state.img_model == "Nano Banana":
            with st.expander("图片上传", expanded=False):
                st.session_state.uploaded_image = st.file_uploader(
                    "上传图片",
                    type=["jpg", "jpeg", "png"]
                )
                if st.session_state.uploaded_image:
                    image = Image.open(st.session_state.uploaded_image)
                    Iwidth, Iheight = image.size
                    if Iwidth > 256 or Iheight > 256:
                        scale = 256 / max(Iheight, Iwidth)
                        new_h, new_w = int(Iheight * scale), int(Iwidth * scale)
                        image = image.resize((new_w, new_h), Image.BILINEAR)
                    st.image(image, caption="图片预览")

    if user_input := st.chat_input("在这里输入您的提示词："):
        messages = Translator(user_input)
        model_parameter = {
            "model": 'deepseek-chat',
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096
        }
        response = st.session_state.openai_client.chat.completions.create(
            stream=False,
            **model_parameter
        )
        prompt = response.choices[0].message.content
        if prompt and st.session_state.uploaded_image:
            with st.spinner('正在生成中,请耐心等待...'):
                if st.session_state.img_model == "Nano Banana":
                    response = st.session_state.google_client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=[prompt, Image.open(st.session_state.uploaded_image).convert('RGB')],
                    )
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            st.session_state.image = Image.open(io.BytesIO(part.inline_data.data))
                else:
                    image_bytes = io.BytesIO()
                    Image.open(st.session_state.uploaded_image).save(image_bytes, format='PNG')
                    image_bytes.seek(0)
                    
                    response = st.session_state.openai_client.images.edit(
                        model=st.session_state.img_model,
                        image=image_bytes,
                        prompt=prompt,
                        size=resulution
                    )
                    st.session_state.image = response.data[0].url

        elif prompt and st.session_state.uploaded_image is None:
            with st.spinner('正在生成中,请耐心等待...'):
                if st.session_state.img_model == "FLUX":
                    img_url = flux_generation(model_name, prompt, width, height)
                    st.session_state.image = Image.open(request.urlopen(img_url))
                elif st.session_state.img_model == "Nano Banana":
                    response = st.session_state.google_client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=[prompt],
                    )
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            st.session_state.image = Image.open(io.BytesIO(part.inline_data.data))
                else:
                    response = st.session_state.openai_client.images.generate(
                        model=st.session_state.img_model,
                        prompt=prompt,
                        size=resulution
                    )
                    st.session_state.image = response.data[0].url
                st.rerun()
        else:
            st.warning("请先填入提示词")


main()
