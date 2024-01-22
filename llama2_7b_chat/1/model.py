import io
import time
import requests
import random
import base64
import ray
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from PIL import Image

import numpy as np

from instill.helpers.const import DataType, TextGenerationChatInput
from instill.helpers.ray_io import (
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    StandardTaskIO,
)

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


from conversation import Conversation, conv_templates, SeparatorStyle


@instill_deployment
class Llama2Chat:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            prefer_safe=True,
        )

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    # async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
    async def __call__(self, req):
        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=req)
        )
        print("----------------________")
        print(task_text_generation_chat_input)
        print("----------------________")

        print("print(task_text_generation_chat.prompt")
        print(task_text_generation_chat_input.prompt)
        print("-------\n")

        print("print(task_text_generation_chat.prompt_images")
        print(task_text_generation_chat_input.prompt_images)
        print("-------\n")

        print("print(task_text_generation_chat.chat_history")
        print(task_text_generation_chat_input.chat_history)
        print("-------\n")

        print("print(task_text_generation_chat.system_message")
        print(task_text_generation_chat_input.system_message)
        print("-------\n")

        print("print(task_text_generation_chat.max_new_tokens")
        print(task_text_generation_chat_input.max_new_tokens)
        print("-------\n")

        print("print(task_text_generation_chat.temperature")
        print(task_text_generation_chat_input.temperature)
        print("-------\n")

        print("print(task_text_generation_chat.top_k")
        print(task_text_generation_chat_input.top_k)
        print("-------\n")

        print("print(task_text_generation_chat.random_seed")
        print(task_text_generation_chat_input.random_seed)
        print("-------\n")

        print("print(task_text_generation_chat.stop_words")
        print(task_text_generation_chat_input.stop_words)
        print("-------\n")

        print("print(task_text_generation_chat.extra_params")
        print(task_text_generation_chat_input.extra_params)
        print("-------\n")

        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)

        # Process chat_history
        # Preprocessing
        prompt_roles = ["USER", "ASSISTANT", "SYSTEM"]
        if (
            task_text_generation_chat_input.chat_history is not None
            and len(task_text_generation_chat_input.chat_history) > 0
        ):
            prompt_conversation = []
            default_system_message = task_text_generation_chat_input.system_message
            for chat_entity in task_text_generation_chat_input.chat_history:
                role = str(chat_entity["role"]).upper()
                chat_history_messages = None
                chat_hisotry_images = []

                for chat_entity_message in chat_entity["content"]:
                    if chat_entity_message["type"] == "text":
                        if chat_history_messages is not None:
                            raise ValueError(
                                "Multiple text message detected"
                                " in a single chat history entity"
                            )
                        # [{'role': 'system', 'content': [{'type': 'text', 'Content': {'Text': "What's in this image?"}}]}]
                        chat_history_messages = chat_entity_message["Content"]["Text"]
                    elif chat_entity_message["type"] == "image_url":
                        # TODO: imeplement image parser in model_backedn
                        # This field is expected to be base64 encoded string
                        IMAGE_BASE64_PREFIX = (
                            "data:image/jpeg;base64,"  # "{base64_image}"
                        )
                        if len(chat_entity_message["Content"]["ImageUrl"]) == 0:
                            continue
                        elif (
                            "promptImageUrl"
                            in chat_entity_message["Content"]["ImageUrl"]["image_url"][
                                "Type"
                            ]
                        ):
                            image = Image.open(
                                io.BytesIO(
                                    requests.get(
                                        chat_entity_message["Content"]["ImageUrl"][
                                            "image_url"
                                        ]["Type"]["promptImageUrl"]
                                    ).content
                                )
                            )
                            chat_hisotry_images.append(image)
                        elif (
                            "promptImageBase64"
                            in chat_entity_message["Content"]["ImageUrl"]["image_url"][
                                "Type"
                            ]
                        ):
                            image_base64_str = chat_entity_message["Content"][
                                "ImageUrl"
                            ]["image_url"]["Type"]["promptImageBase64"]
                            if image_base64_str.startswith(IMAGE_BASE64_PREFIX):
                                image_base64_str = image_base64_str[
                                    IMAGE_BASE64_PREFIX:
                                ]
                            # expected content in url with base64 format:
                            # f"data:image/jpeg;base64,{base64_image}"
                            pil_img = Image.open(
                                io.BytesIO(base64.b64decode(image_base64_str))
                            )
                            image = np.array(pil_img)
                            if len(image.shape) == 2:  # gray image
                                raise ValueError(
                                    f"The chat history image shape with {image.shape} is "
                                    f"not in acceptable"
                                )
                            chat_hisotry_images.append(image)
                    else:
                        raise ValueError(
                            "Unsupported chat_hisotry message type"
                            ", expected eithjer 'text' or 'image_url'"
                            f" but get {chat_entity_message['type']}"
                        )

                # TODO: support image message in chat history
                # self.messages.append([role, message])
                if role not in prompt_roles:
                    raise ValueError(
                        f"Role `{chat_entity['role']}` is not in supported"
                        f"role list ({','.join(prompt_roles)})"
                    )
                elif role == prompt_roles[-1] and default_system_message is not None:
                    raise ValueError(
                        "it's ambiguious to set `system_message` and "
                        f"using role `{prompt_roles[-1]}` simultaneously"
                    )
                elif chat_history_messages is None:
                    raise ValueError(
                        f"No message found in chat_history. {chat_entity_message}"
                    )
                if role == prompt_roles[-1]:
                    default_system_message = chat_history_messages
                else:
                    prompt_conversation.append([role, chat_history_messages])

            if default_system_message is None:
                default_system_message = (
                    "You are a helpful, respectful and honest assistant. "
                    "Always answer as helpfully as possible, while being safe.  "
                    "Your answers should not include any harmful, unethical, racist, "
                    "sexist, toxic, dangerous, or illegal content. Please ensure that "
                    "your responses are socially unbiased and positive in nature. "
                    "If a question does not make any sense, or is not factually coherent, "
                    "explain why instead of answering something not correct. If you don't "
                    "know the answer to a question, please don't share false information."
                )
            conv = Conversation(
                system=default_system_message,
                roles=tuple(prompt_roles[:-1]),
                version="llama_v2",
                messages=prompt_conversation,
                offset=0,
                sep_style=SeparatorStyle.LLAMA_2,
                sep="<s>",
                sep2="</s>",
            )
            conv.append_message(conv.roles[0], task_text_generation_chat_input.prompt)
        else:
            if task_text_generation_chat_input.system_message is not None:
                conv = Conversation(
                    system=task_text_generation_chat_input.system_message,
                    roles=tuple(prompt_roles[:-1]),
                    version="llama_v2",
                    messages=[],
                    offset=0,
                    sep_style=SeparatorStyle.LLAMA_2,
                    sep="<s>",
                    sep2="</s>",
                )
            else:
                conv = conv_templates["llama_2"].copy()
            conv.append_message(conv.roles[0], task_text_generation_chat_input.prompt)

        print("----------------")
        print(f"[DEBUG] Conversation Prompt: \n{conv.get_prompt()}")
        print("----------------")

        # TODO: COMBINED CONSEQUENCE CONVERSATIONS
        # if not prompt_in_conversation:
        # conv = conv_templates["llama_2"].copy()
        # conv.append_message(conv.roles[0], task_text_generation_chat_input.prompt)

        # End of Process chat_history

        t0 = time.time()
        sequences = self.pipeline(
            conv.get_prompt(),
            do_sample=True,
            top_k=task_text_generation_chat_input.top_k,
            temperature=task_text_generation_chat_input.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=task_text_generation_chat_input.max_new_tokens,
            **task_text_generation_chat_input.extra_params,
        )

        print(f"Inference time cost {time.time()-t0}s")

        sequences_generated_texts = [seq["generated_text"] for seq in sequences]

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="output",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                )
            ],
            raw_outputs=sequences_generated_texts,
        )


deployable = InstillDeployable(
    Llama2Chat, model_weight_or_folder_name="Llama-2-7b-chat-hf/", use_gpu=True
)

# # Optional
# deployable.update_max_replicas(2)
# deployable.update_min_replicas(0)
