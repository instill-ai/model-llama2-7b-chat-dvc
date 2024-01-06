---
Task: TextGenerationChat
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

```json
{
    "task_inputs": [
        {
            "text_generation_chat": {
                "prompt": "Yo, what's your name?",
                "chat_history": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Your nams is Tony. A helpful assistant."
                            }
                        ]
                    }
                    ,{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "I like to call you Toro."
                            }
                        ]
                    },
                    {
                        "role": "ASSISTANT",
                        "content": [
                            {
                                "type": "text",
                                "text": "Ok, My name is Toro. What can I help you?"
                            }
                        ]
                    }
                ],
                // "system_message": "You are not a human.", // You can use either chat_history or system_message
                "max_new_tokens": "100",
                "temperature": "0.8",
                "top_k": "10",
                "seed": "42"
                // ,"extra_params": {
                //     "test_param_string": "test_param_string_value",
                //     "test_param_int": 123,
                //     "test_param_float": 0.2,
                //     "test_param_arr": [1, 2, 3],
                //     "test_param_onject": { "some_key": "some_value" }
                // }
            }
        }
    ]
}
```
