import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# ✅ Load Fine-Tuned Model
fine_tuned_model_path = "checkpoints/qwen2.5-epoch5"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(fine_tuned_model_path)
processor = Qwen2_5_VLProcessor.from_pretrained(fine_tuned_model_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

def run_inference(image_path, text_prompt="Extract JSON data from this image"):
    conversation = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": text_prompt}]}]
    inputs = processor(text=[text_prompt], return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return output_text[0]

# ✅ Example Usage
image_path = "eg.jpg"
output = run_inference(image_path)
print("📝 Extracted JSON:", output)
