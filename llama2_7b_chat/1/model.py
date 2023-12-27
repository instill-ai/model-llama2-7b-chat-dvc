import random

import ray
import torch
import transformers
from transformers import AutoTokenizer
import numpy as np

from instill.helpers.const import DataType, TextGenerationChatInput
from instill.helpers.ray_io import StandardTaskIO
from instill.helpers.ray_config import (
    instill_deployment,
    InstillDeployable,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)

import vllm

import struct
import io
import json
import struct
from json.decoder import JSONDecodeError

import numpy as np
from PIL import Image


def deserialize_bytes_tensor(encoded_tensor):
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects

    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in 'C' order.

    """
    strs = []
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from(f"<{l}s", val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return np.array(strs, dtype=bytes)


@instill_deployment
class Llama2Chat:
    def __init__(self, model_path: str):
        # self.application_name = "_".join(model_path.split("/")[3:5])
        # self.deployement_name = model_path.split("/")[4]
        # self.llm_engine = LLM(
        #     model=model_path,
        #     gpu_memory_utilization=0.95,
        #     tensor_parallel_size=1,
        # )
        print(f"This is a Dummy Llama2Chat with path {model_path}")
        print(f"vllm version: {vllm.__version__}")
        print(f"ray version: {ray.__version__}")
        # print(f"transformers version: {transformers.__version__}")
        # print(f"np version: {np.__version__}")

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="python",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        resp = ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[],
            raw_output_contents=[],
        )
        print("???????????????????????????")
        print(request)
        from typing import Any, Dict, List, Union

        class TextGenerationChatInput:
            prompt = ""
            prompt_images: Union[List[np.ndarray], None] = None
            chat_history: Union[List[str], None] = None
            system_message: Union[str, None] = None
            max_new_tokens = 100
            temperature = 0.8
            top_k = 1
            random_seed = 0
            stop_words: Any = ""  # Optional
            extra_params: Dict[str, str] = {}

        text_generation_chat_input = TextGenerationChatInput()
        for i, b_input_tensor in zip(request.inputs, request.raw_input_contents):
            input_name = i.name

            print("..... parse: ", input_name)
            print(text_generation_chat_input)
            print("...")
            if input_name == "prompt":
                input_tensor = deserialize_bytes_tensor(b_input_tensor)
                text_generation_chat_input.prompt = str(input_tensor[0].decode("utf-8"))
                print(
                    f"[DEBUG] input `prompt` type\
                        ({type(text_generation_chat_input.prompt)}): {text_generation_chat_input.prompt}"
                )

            if input_name == "prompt_images":
                input_tensors = deserialize_bytes_tensor(b_input_tensor)
                images = []
                print("..... inside prompt_images, input_tensors: ", input_tensors)
                for enc in input_tensors:
                    print("..... inside prompt_images, enc: ", enc)
                    if len(enc) == 0:
                        continue  # of using try - catch (But I think catch is better)
                    else:
                        trimed_enc = enc[2:-2]
                        # model-backend encoede like:
                        # enc:  b'["/9j "]'"
                    if not type(trimed_enc) == bytes:
                        trimed_enc = trimed_enc.astype(bytes)
                    pil_img = Image.open(io.BytesIO(trimed_enc))  # RGB
                    image = np.array(pil_img)
                    if len(image.shape) == 2:  # gray image
                        raise ValueError(
                            f"The image shape with {image.shape} is "
                            f"not in acceptable"
                        )
                    images.append(image)
                # TODO: check wethere there are issues in batch size dimention
                text_generation_chat_input.prompt_images = images
                print(
                    "[DEBUG] input `prompt_images` type"
                    f"({type(text_generation_chat_input.prompt_images)}): "
                    f"{text_generation_chat_input.prompt_images}"
                )

            if input_name == "chat_history":
                input_tensor = deserialize_bytes_tensor(b_input_tensor)
                chat_history_str = str(input_tensor[0].decode("utf-8"))
                print(
                    "[DEBUG] input `chat_history_str` type"
                    f"({type(chat_history_str)}): "
                    f"{chat_history_str}"
                )
                try:
                    text_generation_chat_input.chat_history = json.loads(
                        chat_history_str
                    )
                except JSONDecodeError:
                    print("[DEBUG] WARNING `extra_params` parsing faield!")
                    continue

            if input_name == "system_message":
                input_tensor = deserialize_bytes_tensor(b_input_tensor)
                text_generation_chat_input.system_message = str(
                    input_tensor[0].decode("utf-8")
                )
                print(
                    "[DEBUG] input `system_message` type"
                    f"({type(text_generation_chat_input.system_message)}): "
                    f"{text_generation_chat_input.system_message}"
                )

            if input_name == "max_new_tokens":
                text_generation_chat_input.max_new_tokens = int.from_bytes(
                    b_input_tensor, "little"
                )
                print(
                    "[DEBUG] input `max_new_tokens` type"
                    f"({type(text_generation_chat_input.max_new_tokens)}): "
                    f"{text_generation_chat_input.max_new_tokens}"
                )

            if input_name == "top_k":
                text_generation_chat_input.top_k = int.from_bytes(
                    b_input_tensor, "little"
                )
                print(
                    "[DEBUG] input `top_k` type"
                    f"({type(text_generation_chat_input.top_k)}): "
                    f"{text_generation_chat_input.top_k}"
                )

            if input_name == "temperature":
                text_generation_chat_input.temperature = struct.unpack(
                    "f", b_input_tensor
                )[0]
                print(
                    "[DEBUG] input `temperature` type"
                    f"({type(text_generation_chat_input.temperature)}): "
                    f"{text_generation_chat_input.temperature}"
                )
                text_generation_chat_input.temperature = round(
                    text_generation_chat_input.temperature, 2
                )

            if input_name == "random_seed":
                text_generation_chat_input.random_seed = int.from_bytes(
                    b_input_tensor, "little"
                )
                print(
                    "[DEBUG] input `random_seed` type"
                    f"({type(text_generation_chat_input.random_seed)}): "
                    f"{text_generation_chat_input.random_seed}"
                )

            if input_name == "extra_params":
                input_tensor = deserialize_bytes_tensor(b_input_tensor)
                extra_params_str = str(input_tensor[0].decode("utf-8"))
                print(
                    "[DEBUG] input `extra_params` type"
                    f"({type(extra_params_str)}): "
                    f"{extra_params_str}"
                )

                try:
                    text_generation_chat_input.extra_params = json.loads(
                        extra_params_str
                    )
                except JSONDecodeError:
                    print("[DEBUG] WARNING `extra_params` parsing faield!")
                    continue

        print("???????????????????????????")
        # task_text_generation_chat_input: TextGenerationChatInput = (
        #     StandardTaskIO.parse_task_text_generation_chat_input(request=request)
        # )
        task_text_generation_chat_input = text_generation_chat_input
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

        print("print(task_text_generation_chat.extra_params")
        print(task_text_generation_chat_input.extra_params)
        print("-------\n")

        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)
            # torch.manual_seed(task_text_generation_chat_input.random_seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(task_text_generation_chat_input.random_seed)

        # Handle Prompt
        # prompt = task_text_generation_chat_input.conversation
        # prompt_in_conversation = False
        # try:
        # parsed_conversation = json.loads(prompt)
        parsed_conversation = task_text_generation_chat_input.conversation
        # turn in to converstation?

        # using fixed roles
        roles = ["USER", "ASSISTANT"]
        roles_lookup = {x: i for i, x in enumerate(roles)}

        sequences = []

        sequences.append(
            {"generated_text": "This is a dummy model response"}  # .encode("utf-8")
        )

        task_text_generation_chat_output = (
            StandardTaskIO.parse_task_text_generation_chat_output(sequences=sequences)
        )

        resp.outputs.append(
            InferTensor(
                name="text",
                shape=[1, len(sequences)],
                datatype=str(DataType.TYPE_STRING),
            )
        )

        resp.raw_output_contents.append(task_text_generation_chat_output)

        return resp


deployable = InstillDeployable(
    Llama2Chat, model_weight_or_folder_name="Llama-2-7b-chat-hf/", use_gpu=True
)

deployable.update_max_replicas(2)
deployable.update_min_replicas(0)
