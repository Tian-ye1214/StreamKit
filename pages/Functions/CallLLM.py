import streamlit as st
from openai import AsyncOpenAI
import os
import json


class CallLLM:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))

    async def call(self, reason_placeholder, message_placeholder, stream, **model_params):
        if stream:
            content = ""
            reasoning_content = ""
            async for chunk in await self.client.chat.completions.create(
                    stream=True,
                    **model_params
            ):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if getattr(delta, 'reasoning_content', None):
                        reasoning_content += delta.reasoning_content
                        reason_placeholder.markdown(
                            f"<div style='background:#c2c2c2; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                            f"🤔 {reasoning_content}</div>",
                            unsafe_allow_html=True
                        )
                    if delta and delta.content is not None:
                        content += delta.content
                        message_placeholder.markdown(
                            f"<div style='font-size:16px; margin-top:10px;'>{content}</div>",
                            unsafe_allow_html=True
                        )
            assistant_response = content

        else:
            response = await self.client.chat.completions.create(
                    stream=False,
                    **model_params
            )
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
            assistant_response = response.choices[0].message.content

            if reasoning_content:
                reason_placeholder.markdown(
                    f"<div style='background:#c2c2c2; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                    f"🤔 {reasoning_content}</div>",
                    unsafe_allow_html=True
                )
            message_placeholder.markdown(assistant_response)

        js_escaped_conversation = json.dumps(f"assistant: {assistant_response}")
        copy_script = f"""
<style>
    .copy-btn {{
        border: 1px solid #ccc;
        background-color: #f0f0f0;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
    }}
    .copy-btn:hover {{
        background-color: #e0e0e0;
    }}
</style>
<button class="copy-btn" id="copy-btn-unique" onclick="copyToClipboard()">复制当前对话</button>
<span id="copy-message-unique" style="margin-left: 10px; color: green; visibility: hidden;">复制成功</span>

<script>
function copyToClipboard() {{
    const textToCopy = {js_escaped_conversation};
    navigator.clipboard.writeText(textToCopy).then(function() {{
        var copyMessage = document.getElementById('copy-message-unique');
        copyMessage.style.visibility = 'visible';
        setTimeout(function() {{
            copyMessage.style.visibility = 'hidden';
        }}, 2000);
    }}, function(err) {{
        console.error('复制失败: ', err);
        alert('复制失败');
    }});
}}
</script>
"""
        st.components.v1.html(copy_script, height=50)

        return assistant_response
