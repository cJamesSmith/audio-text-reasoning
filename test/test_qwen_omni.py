import soundfile as sf

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch

model = "Qwen/Qwen2.5-Omni-7B"

model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "enable_audio_output": False,
        }

# model = Qwen2_5OmniModel.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     # device_map="balanced_low_0",
#     attn_implementation="flash_attention_2",
#     enable_audio_output=False,
# )
model = Qwen2_5OmniModel.from_pretrained(model, **model_init_kwargs)

print(model.thinker.audio_tower.layers[0].self_attn.q_proj.weight)