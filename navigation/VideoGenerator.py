import requests
import os
import streamlit as st
from PIL import Image
from pages.Functions.ExtractFileContents import encode_image_to_base64
import time
import glob
import random
from openai import OpenAI
from pages.Functions.Prompt import Video


def initialization():
    if "video_submit_url" not in st.session_state:
        st.session_state.video_submit_url = "https://api.siliconflow.cn/v1/video/submit"
    if "video_get_url" not in st.session_state:
        st.session_state.video_get_url = "https://api.siliconflow.cn/v1/video/status"
    if "requestId" not in st.session_state:
        st.session_state.requestId = None
    if "video_url" not in st.session_state:
        st.session_state.video_url = None
    if 'api_key' not in st.session_state:
        key = os.environ.get('SiliconFlow_API_KEY')
        st.session_state.api_key = "Bearer " + key
    if 'image_type' not in st.session_state:
        st.session_state.image_type = None
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None
    if 'client' not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.environ.get("ZhiZz_API_KEY"), base_url=os.environ.get("ZhiZz_URL"))


def submit_video(prompt, resolution, seed, negative_prompt, base64_image=None):
    st.session_state.current_prompt = prompt
    if base64_image:
        image = f"data:{st.session_state.image_type};base64," + base64_image
        payload = {
            "model": "Wan-AI/Wan2.2-I2V-A14B",
            "prompt": str(prompt),
            "image_size": str(resolution),
            "negative_prompt": str(negative_prompt),
            "seed": int(seed),
            "image": str(image),
        }
    else:
        payload = {
            "model": "Wan-AI/Wan2.2-T2V-A14B",
            "prompt": str(prompt),
            "image_size": str(resolution),
            "negative_prompt": str(negative_prompt),
            "seed": int(seed),
        }
    headers = {
        "Authorization": st.session_state.api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(st.session_state.video_submit_url, json=payload, headers=headers)
    if response.status_code == 200:
        st.session_state.requestId = response.json()['requestId']
        st.rerun()
    else:
        st.error(f"Request failed with status code {response.status_code}")
        st.session_state.requestId = None


def get_video(requestId):
    st.info(f"本次生成Id:{requestId}，您可以保存ID，稍后通过ID获取视频。")
    info_placeholder = st.empty()
    payload = {"requestId": requestId}
    headers = {
        "Authorization": st.session_state.api_key,
        "Content-Type": "application/json"
    }
    while True:
        time.sleep(15)
        response = requests.post(st.session_state.video_get_url, json=payload, headers=headers)
        if response.status_code == 200:
            if response.json()['status'] == 'Succeed':
                st.session_state.requestId = None
                return response.json()['results']['videos'][0]['url']
            else:
                info_placeholder.info(f"当前生成状态：{response.json()['status']}")
        else:
            st.error(f"Request failed with status code {response.status_code}。请检查生成ID是否有效:{requestId}")


def main():
    st.markdown("""
    <h1 style='text-align: center;'>
        AI视频生成
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    initialization()
    if st.session_state.current_prompt is not None:
        st.info(f"当前提示词：{st.session_state.current_prompt}")
    if st.session_state.requestId is not None:
        with st.spinner("生成中...(预计需要3-5分钟)"):
            st.session_state.video_url = get_video(st.session_state.requestId)
    resolution_mapping = {
        "16:9": "1280x720",
        "9:16": "720x1280",
        "1:1": "960x960",
    }
    with st.sidebar:
        st.title("生成新视频")
        negative_control_words = """
色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,
画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走",
        """
        resolution_display = st.selectbox("视频分辨率", list(resolution_mapping.keys()), index=0, help="视频分辨率")
        resolution = resolution_mapping[resolution_display]
        seed = st.text_input("随机种子", value=3407,
                             help="决定生成视频的随机性起点，相同种子可复现结果，不同种子产生多样变化。")
        negative_prompt = st.text_input("负面提示词", value=negative_control_words, help="让模型不要生成的元素")
        uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            base64_image = encode_image_to_base64(uploaded_image)
            st.session_state.image_type = uploaded_image.type
            image = Image.open(uploaded_image)
            width, height = image.size
            if width > 256 or height > 256:
                scale = 256 / max(height, width)
                new_h, new_w = int(height * scale), int(width * scale)
                image = image.resize((new_w, new_h), Image.BILINEAR)
            st.image(image, caption="图片预览")
        else:
            base64_image = None
            st.session_state.image_type = None
        st.divider()
        if st.button("不知道写点什么？让AI来点惊喜！"):
            messages = Video()
            model_parameter = {
                "model": 'kimi-k2-0711-preview',
                "messages": messages,
                "temperature": 1.5,
                "top_p": 0.95,
                "max_tokens": 8192
            }
            response = st.session_state.client.chat.completions.create(
                stream=False,
                **model_parameter
            )
            prompt = response.choices[0].message.content
            submit_video(prompt, resolution, seed, negative_prompt, base64_image)
        st.divider()
        st.title("从ID获取视频")
        request_id_input = st.text_input("输入requestId")
        if st.button("获取视频"):
            if request_id_input:
                st.session_state.requestId = request_id_input.strip()
                st.rerun()
    if prompt := st.chat_input("在这里输入视频描述："):
        submit_video(prompt, resolution, seed, negative_prompt, base64_image)

    if st.session_state.video_url is not None:
        st.info("注意保存下载视频哦，视频会在1小时之内删除！")
        st.video(st.session_state.video_url)
    else:
        video_list = glob.glob("static/video/*")
        st.video(random.choice(video_list))


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'video'
current_page = 'video'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
main()
