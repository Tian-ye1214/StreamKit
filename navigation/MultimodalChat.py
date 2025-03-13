# -*- coding: utf-8 -*-
import streamlit as st
from pages.Functions.BackendInteraction import BackendInteractionLogic
from pages.Functions.Constants import VISIONMODAL_MAPPING
import io


def main():
    backend = BackendInteractionLogic()
    backend.initialize_session_state()

    st.markdown("""
    <h1 style='text-align: center;'>
        Multimodal Chat
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **当语言遇见视觉，智能呈现新维度** 🌟

        🖼️ **视觉操作指南**\n
        1. 侧边栏选择你想要的视觉模型
        2. 上传图片或输入文字描述
        3. 调节生成参数获取理想效果

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 20px;">
            🎆 试试这些神奇组合：<br>
            • 上传设计稿让AI提供改进建议<br>
            • 上传数学题让AI提供解题思路<br>
            • 分析X光片并解释医学特征<br>
            每个像素都充满智慧！
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        backend.user_interaction()
        backend.start_new_conversation()

        st.markdown("""
        <h3 style='text-align: center;'>
            模型配置
        </h3>
        """, unsafe_allow_html=True)
        model_display = st.selectbox("选择模型", list(VISIONMODAL_MAPPING.keys()), index=1, help="选择模型")
        st.session_state.model = VISIONMODAL_MAPPING[model_display]

        backend.parameter_configuration()

    backend.image_upload()

    if st.session_state.model == "deepseek-ai/Janus-Pro-1B":
        with st.expander("Janus工作模式选择", expanded=False):
            st.session_state.janus_mode = st.radio(
                "",
                ["图片理解模式", "图片生成模式"],
                index=0,
                horizontal=True,
            )
            if st.session_state.janus_mode == "图片生成模式":
                st.session_state.JanusTemperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1,
                                                         help="控制画面内容多样性，值越高多样性越高")
                st.session_state.Janus_cfg_weight = st.slider("cfg_weight", 5.0, 10.0, 7.5, 0.1,
                                                         help="控制提示词和生成图片的相关性，值越高相关性越高")
        with st.expander("生成图片操作", expanded=False):
            if 'generated_images' in st.session_state:
                st.markdown("### 生成图片预览")
                cols = st.columns(3)
                for idx, img in enumerate(st.session_state.generated_images):
                    with cols[idx % 3]:
                        st.image(img, use_container_width=True)
                        # 创建下载按钮
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label=f"下载图片 {idx + 1}",
                            data=byte_im,
                            file_name=f"generated_image_{idx + 1}.png",
                            mime="image/png",
                            key=f"download_{idx}"
                        )

                # 清空按钮
                if st.button("清空所有生成图片"):
                    del st.session_state.generated_images
                    st.rerun()
            else:
                st.info("暂无生成图片")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("在这里输入您的问题："):
        backend.user_input(prompt)
        backend.search_interaction()

        with st.chat_message("assistant"):
            try:
                if st.session_state.model == "deepseek-ai/Janus-Pro-1B":
                    if st.session_state.get('janus_mode', None) == "图片理解模式" and st.session_state.uploaded_image:
                        if st.session_state.uploaded_image:
                            from pages.Functions.MmConversion import mmconversion
                            assistant_response = mmconversion(st.session_state.model, st.session_state.uploaded_image,
                                                              prompt)
                            st.markdown(assistant_response)
                        else:
                            st.error("请先上传图片！")
                            return
                    elif st.session_state.get('janus_mode', None) == "图片生成模式":
                        from pages.Functions.MmGenerator import mmgeneration
                        generated_images = mmgeneration(st.session_state.model, prompt,
                                                        temperature=st.session_state.get('JanusTemperature', 1.0),
                                                        cfg_weight=st.session_state.get('Janus_cfg_weight', 7.5))
                        if generated_images:
                            st.session_state.generated_images = generated_images  # 存储生成的图片
                            st.rerun()  # 触发页面刷新以显示图片
                else:
                    backend.ai_generation()

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


main()
