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
    get_compose_ray_address,
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

import json
from vllm import SamplingParams, LLM
from conversation import Conversation, conv_templates, SeparatorStyle


ray.init(address=get_compose_ray_address(10001))
# this import must come after `ray.init()`
from ray import serve


@instill_deployment
class Llama2Chat:
    def __init__(self, model_path: str):
        # self.application_name = "_".join(model_path.split("/")[3:5])
        # self.deployement_name = model_path.split("/")[4]
        self.llm_engine = LLM(
            model=model_path,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
        )

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="python",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="conversation",
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

        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=request)
        )

        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)
            # torch.manual_seed(task_text_generation_chat_input.random_seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(task_text_generation_chat_input.random_seed)

        # Handle Prompt
        prompt = task_text_generation_chat_input.conversation
        prompt_in_conversation = False
        try:
            parsed_conversation = json.loads(prompt)
            # turn in to converstation?

            # using fixed roles
            roles = ["USER", "ASSISTANT"]
            roles_lookup = {x: i for i, x in enumerate(roles)}

            conv = None
            for i, x in enumerate(parsed_conversation):
                role = str(x["role"]).upper()
                print(f'[DEBUG] Message {i}: {role}: {x["content"]}')
                if i == 0:
                    if role == "SYSTEM":
                        conv = Conversation(
                            system=str(x["content"]),
                            roles=("USER", "ASSISTANT"),
                            version="llama_v2",
                            messages=[],
                            offset=0,
                            sep_style=SeparatorStyle.LLAMA_2,
                            sep="<s>",
                            sep2="</s>",
                        )
                    else:
                        conv = conv_templates["llama_2"].copy()
                        conv.roles = tuple(roles)
                        conv.append_message(
                            conv.roles[roles_lookup[role]], x["content"]
                        )
                else:
                    conv.append_message(conv.roles[roles_lookup[role]], x["content"])
            prompt_in_conversation = True
        except json.decoder.JSONDecodeError:
            pass

        if not prompt_in_conversation:
            conv = conv_templates["llama_2"].copy()
            conv.append_message(conv.roles[0], prompt)

        sampling_params = SamplingParams(
            temperature=task_text_generation_chat_input.temperature,
            max_tokens=task_text_generation_chat_input.max_new_tokens,
            top_k=task_text_generation_chat_input.top_k
            # **extra_params,
        )

        vllm_outputs = self.llm_engine.generate(conv.get_prompt(), sampling_params)

        sequences = []
        for vllm_output in vllm_outputs:
            # concated_complete_output = prompt + "".join([ # Chat model no needs to repeated the prompt
            concated_complete_output = "".join(
                [str(complete_output.text) for complete_output in vllm_output.outputs]
            )
            sequences.append(
                {"generated_text": concated_complete_output.strip()}  # .encode("utf-8")
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
    Llama2Chat, model_weight_or_folder_name="Llama-2-7b-chat-hf/"
)

# you can also have a fine-grained control of the cpu and gpu resources allocation
deployable.update_num_cpus(4)
deployable.update_num_gpus(1)
