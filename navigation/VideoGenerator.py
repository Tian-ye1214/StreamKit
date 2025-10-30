import requests
import os
import streamlit as st
import time
import glob
import random
from openai import OpenAI
from pages.Functions.Prompt import Video


def initialization():
    if "video_url" not in st.session_state:
        st.session_state.video_url = "https://api.zhizengzeng.com/v1/videos"
    if "requestId" not in st.session_state:
        st.session_state.requestId = None
    if "video" not in st.session_state:
        st.session_state.video = None
    if 'api_key' not in st.session_state:
        key = os.environ.get('ZhiZz_API_KEY')
        st.session_state.api_key = "Bearer " + key
    if 'image_type' not in st.session_state:
        st.session_state.image_type = None
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None
    if 'client' not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.environ.get("ZhiZz_API_KEY"), base_url=os.environ.get("ZhiZz_URL"))


def submit_video(prompt, resolution, seconds):
    st.session_state.current_prompt = prompt
    headers = {
        "Authorization": st.session_state.api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sora-2",
        "prompt": str(prompt),
        "size": str(resolution),
        "seconds": str(seconds),
    }
    response = requests.post(st.session_state.video_url, json=payload, headers=headers)

    if response.status_code == 200:
        st.session_state.requestId = response.json()['id']
        st.rerun()
    else:
        st.error(f"Request failed with status code {response.status_code}")
        st.session_state.requestId = None


def get_video(video_id, variant="video"):
    st.info(f"本次生成Id:{video_id}，您可以保存ID，稍后通过ID获取视频。")
    headers = {
        "Authorization": st.session_state.api_key,
        "Accept": "application/binary",
    }
    params = {"variant": variant} if variant else {}
    while True:
        time.sleep(30)
        try:
            response = requests.get(
                f"{st.session_state.video_url}/{video_id}",
                headers=headers,
                params=params,
                timeout=300
            )
            if response.json()['status'] == 'completed':
                response = requests.get(
                    f"{st.session_state.video_url}/{video_id}/content",
                    headers=headers,
                    params=params,
                    timeout=300
                )
                return response.content
            else:
                st.info(f"视频正在生成中...状态：{response.json()['status']}")
        except Exception as e:
            st.error(f"下载出错: {str(e)}")
            return None


def main():
    st.markdown("""
    <h1 style='text-align: center;'>
        Sora2视频生成
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    initialization()
    if st.session_state.current_prompt is not None:
        st.info(f"当前提示词：{st.session_state.current_prompt}")
    if st.session_state.requestId is not None:
        with st.spinner("生成中...(预计需要3-5分钟)"):
            st.session_state.video = get_video(st.session_state.requestId)
    resolution_mapping = {
        "16:9": "1280x720",
        "9:16": "720x1280",
        "1:1": "960x960",
    }
    with st.sidebar:
        st.title("生成新视频")
        resolution_display = st.selectbox("视频分辨率", list(resolution_mapping.keys()), index=0, help="视频分辨率")
        resolution = resolution_mapping[resolution_display]
        seconds = st.slider("生成秒数", 1, 12, 8, 1)
        st.divider()
        if st.button("不知道写点什么？让AI来点惊喜！"):
            messages = Video()
            model_parameter = {
                "model": 'kimi-k2-0711-preview',
                "messages": messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 8192
            }
            response = st.session_state.client.chat.completions.create(
                stream=False,
                **model_parameter
            )
            prompt = response.choices[0].message.content
            submit_video(prompt, resolution, seconds)
        st.divider()
        st.title("从ID获取视频")
        request_id_input = st.text_input("输入requestId")
        if st.button("获取视频"):
            if request_id_input:
                st.session_state.requestId = request_id_input.strip()
                st.rerun()
    if prompt := st.chat_input("在这里输入视频描述："):
        submit_video(prompt, resolution, seconds)

    if st.session_state.video is not None:
        st.info("注意保存下载视频哦，视频会在1小时之内删除！")
        st.video(st.session_state.video)
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
