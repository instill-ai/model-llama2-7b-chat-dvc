---
Task: TextGenerationCHat
Tags:
  - TextGenerationChat
  - Llama2-7b-chat
---

# Model-llama2-7b-chat-dvc

🔥🔥🔥 Deploy [llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model and on [VDP](https://github.com/instill-ai/vdp).

This repository contains the Llama2-7b Text Completion Generation Model in the [vLLM](https://github.com/vllm-project/vllm) format, managed using [DVC](https://dvc.org/). For information about available extra parameters, please refer to the documentation on [SamplingParams](https://github.com/vllm-project/vllm/blob/v0.2.0/vllm/sampling_params.py) in the vLLM library.

Notes:

- Disk Space Requirements: 13G
- GPU Memory Requirements: 14G

```
{
    "task_inputs": [
        {
            "text_generation": {
                "prompt": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},{\"role\": \"user\", \"content\": \"Who won the world series in 2020?\"},{\"role\": \"assistant\", \"content\": \"The Los Angeles Dodgers won the World Series in 2020.\"},{\"role\": \"user\", \"content\": \"Where was it played?\"}]",
                "max_new_tokens": "100",
                "temperature": "0.8",
                "top_k": "20",
                "random_seed": "0",
                "extra_params": "{\"top_p\": 0.8, \"frequency_penalty\": 1.2}"
            }
        }
    ]
}
```