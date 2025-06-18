import streamlit as st
from openai import AsyncOpenAI
import os


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
                            f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
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
                    f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                    f"🤔 {reasoning_content}</div>",
                    unsafe_allow_html=True
                )
            message_placeholder.markdown(assistant_response)

        copy_script = f"""
            <div id="copy-container-{id(assistant_response)}" style="display:inline;">
                <button onclick="copyToClipboard{id(assistant_response)}()" 
                        style="margin-left:10px; background:#f0f0f0; border:none; border-radius:3px; padding:2px 8px;"
                        title="复制内容">
                    📋
                </button>
                <div id="copy-content-{id(assistant_response)}" style="display:none; white-space: pre-wrap;">{assistant_response.lstrip()}</div>
            </div>
            <script>
                function copyToClipboard{id(assistant_response)}() {{
                    const content = document.getElementById('copy-content-{id(assistant_response)}').innerText;
                    navigator.clipboard.writeText(content);
                    const btn = event.target;
                    btn.innerHTML = '✅';
                    setTimeout(() => {{ btn.innerHTML = '📋'; }}, 500);
                }}
            </script>
            """
        st.components.v1.html(copy_script, height=30)

        return assistant_response
