
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


# video_path = './VideoLLaMA3/assets/cat_and_chicken.mp4'
# question = "What is the cat doing?"

video_path = './VideoLLaMA3/assets/margaritas.mp4'
# question = "How many different drinks are made in the video?"
# question = "What are the different drinks made in the video?"
question = "What is done in the video? Give a detailed answer."

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            # {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 1800}},
            {"type": "text", "text": question},
        ]
    },
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=1024)

response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print(response)

