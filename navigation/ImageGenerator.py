import os
import requests
import time
import torch
import streamlit as st
from urllib import request
from PIL import Image
from diffusers import StableDiffusionXLPipeline

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

lora_path = "/home/li/桌面/ComfyUI/models/loras"
sdxl_path = "/home/li/桌面/datasets_and_models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"

lora_mapping = {
    "古建筑-正面": "Front_Building.safetensors",
    "3D渲染": "3Disometric_rendering.safetensors",
    "现代建筑渲染": "Architecture_rendering.safetensors",
    "轴测图Axonometric": "Axonometric_drawing.safetensors",
    "概念草图": "Concept_sketch.safetensors",
    "未来风格": "Futuristics.safetensors",
    "插画风格": "Illustration.safetensors",
    "日本动漫风格": "Japanese_anime.safetensors",
    "白雪石": "baixueshi.safetensors",
    "毕加索": "Camille Pissarro.safetensors", 
    "华岩": "huayan.safetensors",
    "梵高": "vg-webui.safetensors",
    "山水画": "aligned_ancient_style.safetensors",
    "石涛": "shitao.safetensors",
    "宋徽宗": "songhuizong.safetensors"
}

def diffusion_generation(model,prompt,height,width,num_inference_steps,guidance_scale,batch_size,lora=None):
    if model == "SDXL":
        pipe = StableDiffusionXLPipeline.from_single_file(sdxl_path, torch_dtype=torch.float16)
        if isinstance(lora, str) and lora is not None:
            lora_safetensors_path = os.path.join(lora_path,lora_mapping[lora])
            pipe.load_lora_weights(lora_safetensors_path)
        if isinstance(lora,list) and lora is not None:
            lora_scales=[0.9,0.7]
            for (x,scale) in zip(lora,lora_scales):
                lora_safetensors_path = os.path.join(lora_path,lora_mapping[x])
                pipe.load_lora_weights(lora_safetensors_path)
                pipe.fuse_lora(lora_scale = scale)
        pipe = pipe.to("cuda")
        image = pipe(prompt,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,batch_size=batch_size).images[0]
        st.image(image,width=512)
        return image

def flux_generation(model_name, prompt, width, height):
    # flux_url = os.environ["FLUX_URL"]
    flux_url = os.path.join(os.environ["FLUX_URL"],model_name)

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
        time.sleep(0.5)
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
            st.markdown(f"Result: {result['result']['sample']}")
            img_url = result['result']['sample']
            st.image(img_url)
            # break
            return img_url
        else:
            st.markdown(f"Status: {result['status']}")


def main():
    st.set_page_config(layout="wide")
    st.markdown("""
    <h1 style='text-align: center;'>
        Image Generation
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    
    if 'img_model' not in st.session_state:
        st.session_state.img_model = None
    if 'lora' not in st.session_state:
        st.session_state.lora = None

    with st.sidebar:
        st.subheader("选择模型")
        model = st.selectbox(
            "模型",
            options=["FLUX","SDXL"],
            # options=["FLUX"],
            help="选择图像生成模型"
        )
        if model != st.session_state.img_model:
            st.session_state.img_model = model
        
        st.subheader("配置参数")
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input(
                label = '图像宽度',
                min_value=0, 
                max_value=1536, 
                value=1024, 
                step=1, 
                help='请输入图像宽度'
            )
            if model == "SDXL":
                # batch_size = st.number_input(
                #     label = 'batch size',
                #     min_value=1, 
                #     max_value=4, 
                #     value=1, 
                #     step=1
                # )

                num_inference_steps = st.number_input(
                    label = 'inference steps',
                    min_value=0, 
                    max_value=50, 
                    value=30, 
                    step=10, 
                    help='推理步数' 
            )
        with col2:
            height = st.number_input(
                label = '图像高度',
                min_value=0, 
                max_value=1024, 
                value=1024, 
                step=1, 
                help='请输入图像高度'
            )
            if model == "SDXL":
                guidance_scale = st.slider("guidance_scale", 1.0, 10.0, 7.5, help="guidance_scale")
                # sampler = st.selectbox("采样器")
            
        if model == "SDXL":
            st.subheader("选择风格lora")
            genre = st.radio(
                "选择风格lora",
                ["无","建筑风格", "自定义训练风格"],
                captions=[
                    "SDXL基础模型",
                    "古建筑+七种常见建筑风格",
                    "梵高&山水画",
                ],
                label_visibility="hidden"
            )
            
            if genre == "建筑风格":
                lora = st.multiselect(
                    "建筑风格",
                    options=["古建筑-正面","3D渲染","现代建筑渲染","轴测图Axonometric","概念草图","未来风格","插画风格","日本动漫风格"],
                    default=["古建筑-正面","日本动漫风格"],
                    max_selections = 2
                )
                st.session_state.lora = lora

            # if genre == "绘画风格":
            #     lora = st.selectbox(
            #         "绘画风格",
            #         options=["石涛","白雪石","华岩"],
            #         index = 0
            #     )
            #     st.session_state.lora = lora

            if genre == "自定义训练风格":
                lora = st.selectbox(
                    "自定义训练风格",
                    options=["梵高", "山水画"],
                    index = 0
                )
                st.session_state.lora = lora

        if model == "FLUX":
            st.subheader("选择类型")
            model_name = st.selectbox(
                "type",
                options=["flux-pro-1.1", "flux-pro", "flux-dev"],
                index=2
            )
            
       
    #主界面
    prompt = st.text_area("positive prompt",max_chars=500, placeholder="请输入提示词")
    # negative_prompt = st.text_area("negative prompt",max_chars=60, placeholder="请输入反向提示词")
    
    
    #开始生成
    if st.button("提交"):
        if prompt is not None:  
            with st.spinner('正在生成中,请耐心等待...'):
                if model == "SDXL":
                    image = diffusion_generation(
                        model=model,
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        batch_size=1,
                        lora=st.session_state.lora
                        )
                if model == "FLUX":
                    img_url = flux_generation(model_name, prompt, width, height)
                    image = Image.open(request.urlopen(img_url))

                # st.download_button(
                #     label="下载图片",
                #     data=image,
                #     file_name="img.png",
                #     mime="image/jpeg",
                #     use_container_width=True
                # )
        else:
            st.warning("请先填入提示词")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'ImageGenerator'
current_page = 'ImageGenerator'
if current_page != st.session_state.previous_page:
        st.session_state.clear()
        st.session_state.previous_page = current_page
main()
