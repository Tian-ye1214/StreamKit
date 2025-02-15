import io
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def mmconversion(model_path, image, question):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": "",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = [Image.open(io.BytesIO(image.getvalue())).convert("RGB")]
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
