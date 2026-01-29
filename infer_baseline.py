import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "/qwen_ft/model/qwen25_vl"  #修改为对应的路径
image_path = "/qwen_ft/assets/images/demo.jpg"  #修改为对应的路径

try:
    print("Trying to load with AutoModelForVision2Seq...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    ).eval()
except ValueError as e:
    print(f"AutoModelForVision2Seq failed: {e}")
    print("Trying fallback to AutoModel (assuming it maps to CausalLM)...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    ).eval()

    # 检查有没有 generate 方法
    if not hasattr(model, 'generate'):
        raise RuntimeError(
            f"Loaded model type {type(model)} does not have a 'generate' method.\n"
            "Please check config.json's auto_map. It likely maps AutoModel to the base model instead of the generation model."
        )

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "这是什么?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
)
inputs = inputs.to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
response1 = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
    0]
print(f"Answer: {response1}")