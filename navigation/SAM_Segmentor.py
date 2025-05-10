# -*- coding: utf-8 -*-
import asyncio

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
from pages.SAM2_1.SAM import SAM2Segment
from PIL import Image, ImageDraw
import os
import sys
import io
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


async def initialization():
    if "dino" not in st.session_state:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "pages/ModelCheckpoint/GroundingDINO-T")
        st.session_state.dino_processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        st.session_state.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to("cuda")
    if "coordinates" not in st.session_state:
        st.session_state["coordinates"] = None
    if "clicks" not in st.session_state:
        st.session_state.clicks = []
    if "mask_image" not in st.session_state:
        st.session_state.mask_image = None
        st.session_state.combine_image = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
        st.session_state.latest_masks = None
    if "current_marker" not in st.session_state:
        st.session_state.current_marker = 1
    if "SAM2" not in st.session_state:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sam2_module_path = os.path.join(base_dir, "pages/SAM2_1")
        if sam2_module_path not in sys.path:
            sys.path.append(sam2_module_path)
        sam2_checkpoint = "./pages/SAM2_1/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        st.session_state.SAM2 = SAM2Segment(sam2_checkpoint, model_cfg)
        st.session_state.input_point = []
        st.session_state.input_label = []
        st.session_state.box_input = []
    if "box_coordinates" not in st.session_state:
        st.session_state.box_coordinates = None


async def get_rectangle_coords(
        points: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int]:
    point1, point2 = points
    minx = min(point1[0], point2[0])
    miny = min(point1[1], point2[1])
    maxx = max(point1[0], point2[0])
    maxy = max(point1[1], point2[1])
    return (
        minx,
        miny,
        maxx,
        maxy,
    )


async def resize_image_if_needed(image):
    """
    如果图像尺寸过大，则使用双线性插值调整大小
    """
    if image is None:
        return None

    h, w = image.size[1], image.size[0]
    max_size = 1024

    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized_img = image.resize((new_w, new_h), Image.BILINEAR)

        st.info(f"图像已从 {w}x{h} 调整为 {new_w}x{new_h} 以获得最佳性能")
        return np.array(resized_img)

    return np.array(image)


async def point_inference():
    if st.session_state.current_image is not None:
        st.markdown("### 点击掩码生成")
        if st.session_state.latest_masks is not None:
            masked_image = await st.session_state.SAM2.show_mask(st.session_state.latest_masks,
                                                                 image=st.session_state.current_image)
        else:
            masked_image = st.session_state.current_image
        display_image = await st.session_state.SAM2.show_points(masked_image, st.session_state.clicks)
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

                actual_x = max(0, min(w - 1, actual_x))
                actual_y = max(0, min(h - 1, actual_y))

                click_data = {
                    "x": actual_x,
                    "y": actual_y,
                    "marker": st.session_state.current_marker
                }
                st.session_state.input_point.append([actual_x, actual_y])
                st.session_state.input_label.append(st.session_state.current_marker)

                masks = await st.session_state.SAM2.point_and_box_inference(st.session_state.current_image,
                                                                            np.array(st.session_state.input_point),
                                                                            np.array(st.session_state.input_label),
                                                                            None)
                st.session_state.latest_masks = (masks[0] * 255)
                st.session_state.masks_image = Image.fromarray(st.session_state.latest_masks.astype(np.uint8))
                st.session_state.combine_image = Image.fromarray(
                    await st.session_state.SAM2.show_mask(
                        st.session_state.latest_masks, image=st.session_state.current_image
                    )
                )
                if click_data not in st.session_state.clicks:
                    st.session_state.clicks.append(click_data)
                    st.session_state.last_coord = coords
                    st.rerun()
        except KeyError as e:
            st.error(f"生成分割内容出错: {str(e)}")


async def box_inference():
    if st.session_state.current_image is not None:
        if st.session_state.combine_image:
            masked_image = np.array(st.session_state.combine_image)
        else:
            masked_image = st.session_state.current_image
        img = Image.fromarray(masked_image)
        draw = ImageDraw.Draw(img)

        if st.session_state.box_coordinates:
            coords = await get_rectangle_coords(st.session_state.box_coordinates)
            draw.rectangle(coords, fill=None, outline="red", width=2)

        st.markdown("### 框选目标区域")
        value = streamlit_image_coordinates(
            img,
            key="box_select",
            click_and_drag=True,
            height=img.height,
            use_column_width=False
        )

        if value is not None:
            point1 = (value["x1"], value["y1"])
            point2 = (value["x2"], value["y2"])

            if (point1[0] != point2[0] and point1[1] != point2[1] and
                    st.session_state.box_coordinates != (point1, point2)):
                st.session_state.box_coordinates = (point1, point2)

                box_coords = await get_rectangle_coords(st.session_state.box_coordinates)

                try:
                    with st.spinner("正在生成分割结果..."):
                        st.session_state.box_input.append([box_coords[0], box_coords[1], box_coords[2], box_coords[3]])

                        masks = await st.session_state.SAM2.point_and_box_inference(st.session_state.current_image,
                                                                                    None,
                                                                                    None,
                                                                                    np.array(
                                                                                        st.session_state.box_input))
                        if len(masks.shape) == 4:
                            masks = masks[-1]
                            set_image = np.array(st.session_state.combine_image)
                            st.session_state.latest_masks = (masks[0] * 255)
                            st.session_state.masks_image = Image.fromarray(
                                st.session_state.latest_masks.astype(np.uint8) + np.array(st.session_state.masks_image)
                            )
                        else:
                            set_image = st.session_state.current_image
                            st.session_state.latest_masks = (masks[0] * 255)
                            st.session_state.masks_image = Image.fromarray(
                                st.session_state.latest_masks.astype(np.uint8))
                        st.session_state.combine_image = Image.fromarray(
                            await st.session_state.SAM2.show_mask(
                                st.session_state.latest_masks, image=set_image
                            )
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"框选分割出错: {str(e)}")


async def auto_masks_generator():
    auto_masks = None
    try:
        if st.session_state.current_image is not None:
            st.markdown("### 自动掩码生成")
            if st.session_state.combine_image is not None:
                st.image(st.session_state.combine_image, use_container_width=True, caption="分割图像")
            else:
                st.image(st.session_state.current_image, use_container_width=True, caption="原始图像")

            if st.button("生成全图掩码", help="自动生成全图所有物体的掩码"):
                with st.spinner("正在生成全图掩码..."):
                    auto_masks = await st.session_state.SAM2.auto_mask_genarator(st.session_state.current_image)

            if auto_masks is not None:
                combined_mask = await st.session_state.SAM2.show_masks(st.session_state.current_image, auto_masks)
                st.session_state.masks_image = Image.fromarray(combined_mask)
                blended = cv2.addWeighted(st.session_state.current_image, 0.8,
                                          combined_mask[..., :3], 0.8, 0)
                st.session_state.combine_image = Image.fromarray(blended)
                st.rerun()
    except KeyError as e:
        st.error(f"生成分割内容出错: {str(e)}")


async def inference_with_nature_language():
    box_list = []
    if st.session_state.current_image is not None:
        if st.session_state.combine_image:
            masked_image = st.session_state.combine_image
        else:
            masked_image = Image.fromarray(st.session_state.current_image)
        st.image(masked_image)
        if text_labels := st.chat_input("在这里输入想要分割的地方："):
            if not all(ord(c) < 128 for c in text_labels):
                st.error('目前模型只支持英文输入🥲')
                return
            text_labels = 'a ' + text_labels + '.'
            text_labels = text_labels.lower()
            inputs = st.session_state.dino_processor(images=masked_image, text=text_labels, return_tensors="pt").to(
                "cuda")
            with st.spinner('开始检测'):
                with torch.no_grad():
                    outputs = st.session_state.dino_model(**inputs)

                results = st.session_state.dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[masked_image.size[::-1]]
                )
                result = results[0]
                for box in result["boxes"]:
                    box = [round(x, 2) for x in box.tolist()]
                    box_list.append(box)
                if not box_list:
                    st.error('检测失败')
                    return
                else:
                    st.info(f'检测到目标位置：{box_list}')
            with st.spinner('开始分割'):
                masks = await st.session_state.SAM2.point_and_box_inference(masked_image, input_box=np.array(box_list))
                all_masks = np.zeros(masked_image.size[::-1], dtype=np.float32)
                if len(masks.shape) == 4:
                    for mask in masks:
                        all_masks += mask[0].astype(np.float32)
                else:
                    all_masks = masks[0].astype(np.float32)
                all_masks[all_masks != 0] = 1
                all_masks = (all_masks * 255).astype(np.uint8)
                st.session_state.masks_image = Image.fromarray(all_masks)
                st.session_state.latest_masks = all_masks
                st.session_state.combine_image = Image.fromarray(
                    await st.session_state.SAM2.show_mask(st.session_state.latest_masks, image=np.array(masked_image))
                )
                st.rerun()


async def clear_all():
    st.session_state.clicks = []
    st.session_state.input_point = []
    st.session_state.input_label = []
    st.session_state.latest_masks = None
    st.session_state.combine_image = None
    st.session_state.masks_image = None
    st.session_state.box_coordinates = None


async def main():
    await initialization()
    st.markdown("""
    <h1 style='text-align: center;'>
        SAM2.1交互式语义分割
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **点触之间，精准分离万物** 🌟

        **源项目地址**：https://github.com/facebookresearch/sam2

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

    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        current_file_hash = hash(file_bytes)
        if "previous_file_hash" not in st.session_state or st.session_state.previous_file_hash != current_file_hash:
            await clear_all()
            st.session_state.previous_file_hash = current_file_hash

        st.session_state.current_image = await resize_image_if_needed(Image.open(uploaded_file).convert("RGB"))
    else:
        st.warning('请上传图片')
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ['Point inference', 'Box inference', 'Auto Masks Generation', 'Inference with natural language'])
    with tab1:
        await point_inference()

    with tab2:
        await box_inference()

    with tab3:
        await auto_masks_generator()

    with tab4:
        await inference_with_nature_language()

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
            await clear_all()
            st.rerun()

        if st.session_state.combine_image is not None:
            st.markdown("### 下载分割结果")
            col1, col2 = st.columns(2)

            mask_bytes = io.BytesIO()
            st.session_state.masks_image.save(mask_bytes, format='PNG')
            image_bytes = io.BytesIO()
            st.session_state.combine_image.save(image_bytes, format='PNG')

            with col1:
                st.download_button(
                    label="下载掩码",
                    data=mask_bytes.getvalue(),
                    file_name="mask.png",
                    mime="image/png"
                )

            with col2:
                st.download_button(
                    label="下载带掩码的图像",
                    data=image_bytes.getvalue(),
                    file_name="masked_image.png",
                    mime="image/png"
                )


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'SAM2'
current_page = 'SAM2'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
asyncio.run(main())
