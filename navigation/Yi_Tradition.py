import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import asyncio
from threading import Thread
from openai import AsyncOpenAI
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING
import os
import re
import random
import gc


async def ini_model(model_path):
    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )


async def ini_tokenizer(model_path):
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


async def call_llm(info_placeholder, reason_placeholder, message_placeholder, generated_text, reasoning_content):
    choicen_model = random.choice(list(HIGHSPEED_MODEL_MAPPING.keys()))
    info_placeholder.markdown(f'选择{choicen_model}执行任务')
    choicen_model = HIGHSPEED_MODEL_MAPPING[choicen_model]
    async for chunk in await st.session_state.client.chat.completions.create(
            model=choicen_model,
            messages=st.session_state.messages,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            presence_penalty=st.session_state.presence_penalty,
            frequency_penalty=st.session_state.frequency_penalty,
            max_tokens=st.session_state.max_tokens,
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
                generated_text += delta.content
                message_placeholder.markdown(
                    f"<div style='font-size:16px; margin-top:10px;'>{generated_text}</div>",
                    unsafe_allow_html=True
                )
    return generated_text


async def unload_model():
    """卸载当前加载的模型和tokenizer"""
    if 'model' in st.session_state:
        del st.session_state.model
    if 'tokenizer' in st.session_state:
        del st.session_state.tokenizer
    if 'model_inputs' in st.session_state:
        del st.session_state.model_inputs
    torch.cuda.empty_cache()
    gc.collect()


async def call_yi(info_placeholder, current_message, generated_text, message_placeholder):
    info_placeholder.markdown(f'已选择{st.session_state.model_display}执行任务')
    st.session_state.text = st.session_state.tokenizer.apply_chat_template(
        current_message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    st.session_state.model_inputs = st.session_state.tokenizer([st.session_state.text],
                                                               return_tensors="pt").to(st.session_state.model.device)
    generate_params = dict(
        inputs=st.session_state.model_inputs.input_ids,
        max_new_tokens=st.session_state.max_tokens,
        top_k=st.session_state.top_k,
        top_p=st.session_state.top_p,
        min_p=0.0,
        temperature=st.session_state.temperature,
        streamer=st.session_state.streamer
    )
    thread = Thread(target=st.session_state.model.generate, kwargs=generate_params)
    thread.start()

    for new_text in st.session_state.streamer:
        generated_text += new_text
        message_placeholder.markdown(generated_text)

    return generated_text


async def ini_message():
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
    if 'client' not in st.session_state:
        st.session_state.client = AsyncOpenAI(api_key=os.environ.get('ZhiZz_API_KEY'),
                                              base_url=os.environ.get('ZhiZz_URL'))
    if 'Yi_message_list' not in st.session_state:
        st.session_state.Yi_message_list = []


def contains_yi_text(text):
    yi_pattern = re.compile(r'[\uA000-\uA48F]')
    text_str = str(text).lower()
    special_keywords = ['介绍自己', '彝', ' Yi ', ' yi ', '你是谁']
    if yi_pattern.search(text) or any(keyword in text_str for keyword in special_keywords):
        return True
    return False


async def parameter_settings():
    MODEL_MAPPING = {
        'Qwen3-8B': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/qwen3-8B',
        'Yi-0.0.1-FullTrain-8B': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/qwen3_sft',
        'Yi-0.0.1-LoRA-8B': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/qwen3_lora_sft',
        'Yi-0.0.2-FullTrain-6000steps-8B': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/Yi-0.0.2-Full-6000',
        'Yi-0.0.2-pro-LoRA': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/Yi-0.0.2-pro-LoRA',
        'Yi-0.0.3-2000-LoRA': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/Yi-0.0.3-LoRA',
        'Yi-0.0.3-6000-LoRA': '/home/li/桌面/models/Qwen3-8B-TrainingCheckpoints/Yi-0.0.3-6000-LoRA',
    }
    with st.sidebar:
        previous_model = st.session_state.get('model_display', None)
        st.session_state.model_display = st.selectbox("选择模型", list(MODEL_MAPPING.keys()), index=6, help="选择模型")
        st.session_state.model_path = MODEL_MAPPING[st.session_state.model_display]
        st.session_state.use_agent = st.toggle("使用Agent", value=True, help="使用Agent决策任务调度")
        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                                                         help="控制模型回答的多样性，值越高表示回复多样性越高")
                st.session_state.presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1,
                                                              help="控制回复主题的多样性性，值越高重复性越低")
                st.session_state.max_tokens = st.number_input("Max Tokens",
                                                              min_value=1,
                                                              max_value=32768,
                                                              value=4096,
                                                              help="生成文本的最大长度")
            with col2:
                st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.8, 0.1,
                                                   help="控制词汇选择的多样性,值越高表示潜在生成词汇越多样")
                st.session_state.top_k = st.slider("Top K", 0, 80, 20, 1,
                                                   help="控制词汇选择的多样性,值越高表示潜在生成词汇越多样")
                st.session_state.frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1,
                                                               help="控制回复中相同词汇重复性，值越高重复性越低")

        if previous_model != st.session_state.model_display or 'tokenizer' not in st.session_state or 'model' not in st.session_state:
            await unload_model()
            try:
                with st.spinner('加载模型中...'):
                    tasks = [ini_model(st.session_state.model_path), ini_tokenizer(st.session_state.model_path)]
                    await asyncio.gather(*tasks)
                    st.session_state.streamer = TextIteratorStreamer(st.session_state.tokenizer,
                                                                     skip_prompt=True, skip_special_tokens=True)
            except Exception as e:
                st.error('模型加载出错：', e)
                return


async def main():
    st.markdown("""
    <h1 style='text-align: center;'>
        彝脉传承
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    tasks = [ini_message(), parameter_settings()]
    await asyncio.gather(*tasks)
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("在这里输入您的问题：", key="chat_input"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        current_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]

        with st.chat_message("assistant"):
            info_placeholder = st.empty()
            reason_placeholder = st.empty()
            message_placeholder = st.empty()
            generated_text = ""
            reasoning_content = ""
            try:
                if st.session_state.use_agent and not contains_yi_text(user_input):
                    generated_text = await call_llm(info_placeholder, reason_placeholder, message_placeholder,
                                                    generated_text, reasoning_content)
                else:
                    with torch.inference_mode():
                        generated_text = await call_yi(info_placeholder, current_message, generated_text,
                                                       message_placeholder)
                st.session_state.messages.append({"role": "assistant", "content": generated_text})
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'Yi_tradition'
current_page = 'Yi_tradition'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    torch.cuda.empty_cache()
    st.session_state.previous_page = current_page
torch.cuda.empty_cache()
gc.collect()
asyncio.run(main())
