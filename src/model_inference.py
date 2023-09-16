import re
import asyncio
import requests
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


class inference:
    def __init__(self) -> None:
        code_path = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
        chat_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
        model_basename = "model"
        use_triton = False

        # self.code_tokenizer = AutoTokenizer.from_pretrained(code_path, use_fast=True)

        # self.code_llama = AutoGPTQForCausalLM.from_quantized(
        #     code_path,
        #     use_safetensors=True,
        #     trust_remote_code=True,
        #     device="cuda:0",
        #     use_triton=use_triton,
        #     quantize_config=None,
        # )

        # self.chat_tokenizer = AutoTokenizer.from_pretrained(chat_path, use_fast=True)

        # self.llama_2 = AutoGPTQForCausalLM.from_quantized(
        #     chat_path,
        #     model_basename=model_basename,
        #     use_safetensors=True,
        #     trust_remote_code=True,
        #     device="cuda:0",
        #     use_triton=use_triton,
        #     quantize_config=None,
        # )

    # async def generate_code(self, prompt: str) -> str:
    #     system_message = "Write code to solve the following coding problem that obeys the constraints and passes the example test cases. \
    #     Please wrap your code answer using '```' and specify what language you used immidiately after it without any spaces:"
    #     prompt_template = f"""
    #     [INST]
    #         {system_message}
    #         {prompt}
    #     [/INST]
    #     """

    #     input_ids = self.code_tokenizer(
    #         prompt_template, return_tensors="pt"
    #     ).input_ids.cuda()
    #     output = self.code_llama.generate(
    #         inputs=input_ids, temperature=0.7, max_new_tokens=512
    #     )
    #     response = self.code_tokenizer.decode(output[0])
    #     response = re.sub(r"<s>|</s>", "", response)

    #     cleaned_response = response.split("[/INST]")[1]
    #     cleaned_response = cleaned_response.strip()
    #     return cleaned_response

    # async def chat(self, prompt: str) -> str:
    #     system_message = """<<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>"""

    #     prompt_template = f"""[INST]{system_message}\n{prompt}[/INST] \n \n"""

    #     input_ids = self.chat_tokenizer(
    #         prompt_template, return_tensors="pt"
    #     ).input_ids.cuda()
    #     output = self.llama_2.generate(
    #         inputs=input_ids, temperature=0.7, max_new_tokens=512
    #     )
    #     response = self.chat_tokenizer.decode(output[0])
    #     response = re.sub(r"<s>|</s>", "", response)

    #     cleaned_response = response.split("[/INST]")[1]
    #     cleaned_response = cleaned_response.strip()
    #     return cleaned_response

    async def ChatGPT(self, prompt: str) -> str:
        api_url = "https://www.cursor.so/api/chat/stream"
        payload = {
            "prompt": '{"history":[{"role":"user","content":"QUESTION"}]}'.replace(
                "QUESTION", prompt
            ),
            "history": [],
        }
        r = requests.post(api_url, json=payload)
        if r.status_code == 200:
            return r.content.decode("utf-8")
        return "Sorry, I couldn't answer that question (network error)"

    async def Llama(self, prompt: str) -> str:
        api_url = "https://www.llama2.ai/api"

        payload = {
            "prompt": f"[INST] {prompt} [/INST]",
            "version": "2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
            "systemPrompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "temperature": 0.75,
            "topP": 0.9,
            "maxTokens": 800,
        }
        r = requests.post(api_url, json=payload)
        if r.status_code == 200:
            return r.content.decode("utf-8")
        return "Sorry, I couldn't answer that question (network error)"


if __name__ == "__main__":
    llama = inference()
    print(asyncio.run(llama.Llama("Tell me about yourself")))
