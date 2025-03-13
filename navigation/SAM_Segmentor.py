# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
from pages.SAM2_1.SAM import SAM2Segment
from PIL import Image

st.markdown("""
<style>
/* 更新容器样式 */
.container {
    max-width: 100%;
    padding: 1rem;
}

/* 优化坐标框样式 */
.coordinates-box {
    background: rgba(255,255,255,0.9);
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.coordinates-box h3 {
    margin-top: 0;
    color: #2c3e50;
    font-size: 1.2em;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 0.5rem;
}

/* 历史记录滚动区域 */
.history-list {
    max-height: 300px;
    overflow-y: auto;
    margin-top: 1rem;
}

/* 响应式布局调整 */
@media (max-width: 768px) {
    .coordinates-box {
        position: static;
        max-width: 100%;
        margin-top: 1rem;
    }
}

/* 点击标记样式 */
.click-point {
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}

.click-point.positive {
    background: #4CAF50;
    border: 2px solid #388E3C;
}

.click-point.negative {
    background: #f44336;
    border: 2px solid #D32F2F;
}

/* 图像容器样式 */
.image-container {
    position: relative;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

def initialization():
    if "clicks" not in st.session_state:
        st.session_state.clicks = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
        st.session_state.history_masks = []
    if "current_marker" not in st.session_state:
        st.session_state.current_marker = 1
    if "SAM2" not in st.session_state:
        sam2_checkpoint = "./pages/SAM2_1/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        st.session_state.SAM2 = SAM2Segment(sam2_checkpoint, model_cfg)
        st.session_state.input_point = []
        st.session_state.input_label = []
    if "display_image" not in st.session_state:
        st.session_state.display_image = None

def main():
    initialization()
    st.markdown("""
    <h1 style='text-align: center;'>
        SAM2.1交互式分割页面
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **点触之间，精准分离万物** 🌟

        🧰 **操作指南**：
        
        1. 上传需要分割的图片
        2. 选择标记类型（正/负标记）
        3. 点击目标区域进行分割
        4. 通过侧边栏实时查看坐标记录
        5. 使用历史记录回溯操作步骤

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 20px;">
            🔬 典型应用场景：<br>
            • 人像前景与背景提取<br>
            • 产品摄影背景分离<br>
            • 遥感图像地物识别<br>
            每次点击都带来精准分割！
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.current_image = np.array(Image.open(uploaded_file).convert("RGB"))
    if st.session_state.current_image is not None:
        if st.session_state.history_masks:
            latest_mask = np.array(st.session_state.history_masks[-1])
            masked_image = st.session_state.SAM2.show_mask(latest_mask, image=st.session_state.current_image)
        else:
            masked_image = st.session_state.current_image.copy()

        display_image = st.session_state.SAM2.show_points(masked_image, st.session_state.clicks)
        st.session_state.display_image = display_image

        try:
            coords = streamlit_image_coordinates(
                display_image,
                key="image",
                height=display_image.shape[0],
                use_column_width=False,
                click_and_drag=False
            )
            if coords and coords != st.session_state.get("last_coord"):
                h, w = st.session_state.current_image.shape[:2]

                actual_x = int(coords["x"])
                actual_y = int(coords["y"])

                actual_x = max(0, min(w-1, actual_x))
                actual_y = max(0, min(h-1, actual_y))

                click_data = {
                    "x": actual_x,
                    "y": actual_y,
                    "marker": st.session_state.current_marker
                }
                st.session_state.input_point.append([actual_x, actual_y])
                st.session_state.input_label.append(st.session_state.current_marker)

                masks = st.session_state.SAM2.point_inference(st.session_state.current_image,
                                                              np.array(st.session_state.input_point),
                                                              np.array(st.session_state.input_label))
                st.session_state.history_masks.append(Image.fromarray(masks[0]).convert("L"))
                if click_data not in st.session_state.clicks:
                    st.session_state.clicks.append(click_data)
                    st.session_state.last_coord = coords
                    st.rerun()
        except KeyError as e:
                st.error(f"生成分割内容出错: {str(e)}")

        with st.sidebar:
            marker_type = st.radio(
                "标记类型",
                ["我想要的区域(1)", "我不想要的区域(0)"],
                index=0 if st.session_state.current_marker == 1 else 1,
                key="marker_selector"
            )
            st.session_state.current_marker = 1 if "1" in marker_type else 0
            st.markdown("""
                <div class="coordinates-box">
                    <h3>📌 坐标信息</h3>
                    <p>最新坐标：<br><strong style="color:#4CAF50">({x}, {y})</strong></p>
                    <p>当前标记：<strong style="color:{color}">[{marker}] {type}</strong></p>
                    <div class="history-list">
                        <p>📚 历史记录（最近10条）：</p>
                        {history}
                    </div>
                </div>
                """.format(
                    x=st.session_state.clicks[-1]["x"] if st.session_state.clicks else "N/A",
                    y=st.session_state.clicks[-1]["y"] if st.session_state.clicks else "N/A",
                    marker=st.session_state.current_marker,
                    color="#4CAF50" if st.session_state.current_marker == 1 else "#f44336",
                    type=marker_type,
                    history="\n".join([
                        f'<div class="history-item" style="color: {"#4CAF50" if c["marker"] == 1 else "#f44336"}">'
                        f'→ ({c["x"]}, {c["y"]}) <small>[{c["marker"]}]</small></div>'
                        for c in reversed(st.session_state.clicks[-10:])
                    ])
            ), unsafe_allow_html=True)
            if st.button("清除所有记录"):
                st.session_state.clicks = []
                st.session_state.input_point = []
                st.session_state.input_label = []
                st.session_state.history_masks = []
                st.rerun()


main()


