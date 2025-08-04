import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np

sys.path.append('/home/li/桌面/')

lora_mapping = {
    # "古建筑-正面": "Front_Building.safetensors",
    "古建筑-正面": "ablation-sam_rag_blip.safetensors",
    "3D渲染": "moxing_style-000008.safetensors",
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
    "宋徽宗": "songhuizong_2-000006.safetensors"
}


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from ComfyUI.main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from ComfyUI.utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    from ComfyUI import execution
    from ComfyUI.nodes import init_extra_nodes

    # import ComfyUI.server
    from ComfyUI import server
    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from ComfyUI.nodes import NODE_CLASS_MAPPINGS


def style_transfer(image, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, batch_size,
                   lora=None):
    lora_num = len(lora)
    import_custom_nodes()
    with torch.inference_mode():
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_10 = controlnetloader.load_controlnet(
            control_net_name="controlnet-union-sdxl-1.0.safetensors"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_11 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        model_node = checkpointloadersimple_11

        if lora_num == 1:
            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloader_23 = loraloader.load_lora(
                lora_name=lora_mapping[lora[0]],
                strength_model=0.6000000000000001,
                strength_clip=1,
                model=get_value_at_index(checkpointloadersimple_11, 0),
                clip=get_value_at_index(checkpointloadersimple_11, 1),
            )
            model_node = loraloader_23

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_12 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(model_node, 1),
        )

        cliptextencode_13 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(model_node, 1)
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_15 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_17 = loadimage.load_image(
            image=image
        )

        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            setunioncontrolnettype_16 = setunioncontrolnettype.set_controlnet_type(
                type="hed/pidi/scribble/ted",
                control_net=get_value_at_index(controlnetloader_10, 0),
            )

            controlnetapplyadvanced_14 = controlnetapplyadvanced.apply_controlnet(
                strength=1,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(cliptextencode_12, 0),
                negative=get_value_at_index(cliptextencode_13, 0),
                control_net=get_value_at_index(setunioncontrolnettype_16, 0),
                image=get_value_at_index(loadimage_17, 0),
                vae=get_value_at_index(checkpointloadersimple_11, 2),
            )

            ksampler_19 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(loraloader_23, 0),
                positive=get_value_at_index(controlnetapplyadvanced_14, 0),
                negative=get_value_at_index(controlnetapplyadvanced_14, 1),
                latent_image=get_value_at_index(emptylatentimage_15, 0),
            )

            vaedecode_20 = vaedecode.decode(
                samples=get_value_at_index(ksampler_19, 0),
                vae=get_value_at_index(checkpointloadersimple_11, 2),
            )

            images = get_value_at_index(vaedecode_20, 0)
            saveimage_21 = saveimage.save_images(
                filename_prefix="st_res_", images=get_value_at_index(vaedecode_20, 0)
            )

            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
