import re

import gradio as gr
import torch
import os
from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import List, Optional, Tuple, Dict
from uuid import uuid4
from modelscope import HubApi
import torchaudio
import sys
sys.path.insert(1, "../cosyvoice")
sys.path.insert(1, "../sensevoice")
sys.path.insert(1, "../cosyvoice/third_party/AcademiCodec")
sys.path.insert(1, "../cosyvoice/third_party/Matcha-TTS")
sys.path.insert(1, "../")
from utils.rich_format_small import format_str_v2
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel

# api = HubApi()
# MS_API_TOKEN = os.environ.get('MS_API_TOKEN')
# api.login(MS_API_TOKEN)
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

cosyvoice = CosyVoice('speech_tts/CosyVoice-300M')
asr_model_name_or_path = "iic/SenseVoiceSmall"
sense_voice_model = AutoModel(model=asr_model_name_or_path,
                  vad_model="fsmn-vad",
                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True, device="cuda:0", remote_code="./sensevoice/model.py")
model_name = "qwen2-72b-instruct"
default_system = """
你是一个中英语翻译机。可以将用户的输入直接翻译为中文或英文，不要有多余的解释和句首句尾的文字，直接给出翻译内容即可。请注意你只是一个智能翻译机，你的任务是对用户的输入进行翻译，不要试图回答用户的问题，不要试图回答用户的问题，不要试图回答用户的问题。
任务分为三个步骤：1. 分析用户想要翻译的内容；2. 用户输入英文，则翻译为中文；输入中文，则翻译为英文；3. 不要有前后缀，只需要直接给出目标语言的标签和翻译结果，标签有：<|zh|>、<|en|>、<|jp|>、<|yue|>、<|ko|>
示例：
输入：苹果怎么说
输出：<|en|>Apple
输入：谢谢
输出：<|en|>thank you
输入：pear
输出：<|zh|>梨
输入：walk
输出：<|zh|>走
输入：你来自哪里
输出：<|en|>where are you from
输入：你是谁
输出：<|en|>who are you
"""

os.makedirs("./tmp", exist_ok=True)

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def clear_session() -> History:
    return '', None, None


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([format_str_v2(q['content']), r['content']])
    return system, history


def model_chat(audio, history: Optional[History]
               ) -> Tuple[str, str, History]:
    if audio is None:
        query = ''
        asr_wav_path = None
    else:
        asr_res = transcribe(audio)
        query, asr_wav_path = asr_res['text'], asr_res['file_path']
    if history is None:
        history = []
    system = default_system
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    print(messages)
    gen = Generation()
    llm_stream = False
    if llm_stream:
        gen = gen.call(
            model_name,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            enable_search=False,
            stream=llm_stream,
        )
    else:
        gen = [gen.call(
            model_name,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            enable_search=False,
            stream=llm_stream
        )]
    processed_tts_text = ""
    punctuation_pattern = r'([!?;。！？])'
    for response in gen:
        if response.status_code == HTTPStatus.OK:
            role = response.output.choices[0].message.role
            response = response.output.choices[0].message.content
            print(f"response: {response}")
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            # 对 processed_tts_text 进行转义处理
            escaped_processed_tts_text = re.escape(processed_tts_text)
            tts_text = re.sub(f"^{escaped_processed_tts_text}", "", response)
            if re.search(punctuation_pattern, tts_text):
                parts = re.split(punctuation_pattern, tts_text)
                if len(parts) > 2 and parts[-1] and llm_stream: # parts[-1]为空说明句子以标点符号结束，没必要截断
                    tts_text = "".join(parts[:-1])
                print(f"processed_tts_text: {processed_tts_text}")
                processed_tts_text += tts_text
                print(f"cur_tts_text: {tts_text}")
                # tts_generator = text_to_speech(tts_text)
                tts_generator = text_to_speech_cross_lingual(tts_text, asr_wav_path)
                for output_audio_path in tts_generator:
                    yield history, output_audio_path, None
        else:
            raise ValueError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    if processed_tts_text == response:
        print("turn end")
    else:
        escaped_processed_tts_text = re.escape(processed_tts_text)
        tts_text = re.sub(f"^{escaped_processed_tts_text}", "", response)
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech_cross_lingual(tts_text, asr_wav_path)
        for output_audio_path in tts_generator:
            yield history, output_audio_path, None
        processed_tts_text += tts_text
        print(f"processed_tts_text: {processed_tts_text}")
        print("turn end")


def transcribe(audio):
    samplerate, data = audio
    file_path = f"./tmp/asr_{uuid4()}.wav"

    torchaudio.save(file_path, torch.from_numpy(data).unsqueeze(0), samplerate)

    res = sense_voice_model.generate(
        input=file_path,
        cache={},
        language="zh",
        text_norm="woitn",
        batch_size_s=0,
        batch_size=1
    )
    text = res[0]['text']
    res_dict = {"file_path": file_path, "text": text}
    print(res_dict)
    return res_dict


def preprocess(text):
    seperators = ['.', '。', '?', '!']
    min_sentence_len = 10
    # split sentence
    seperator_index = [i for i, j in enumerate(text) if j in seperators]
    if len(seperator_index) == 0:
        return [text]
    texts = [text[:seperator_index[i] + 1] if i == 0 else text[seperator_index[i - 1] + 1: seperator_index[i] + 1] for i in range(len(seperator_index))]
    remains = text[seperator_index[-1] + 1:]
    if len(remains) != 0:
        texts.append(remains)
    # merge short sentence
    texts_merge = []
    this_text = texts[0]
    for i in range(1, len(texts)):
        if len(this_text) >= min_sentence_len:
            texts_merge.append(this_text)
            this_text = texts[i]
        else:
            this_text += texts[i]
    texts_merge.append(this_text)
    return texts


def text_to_speech_cross_lingual(text, audio_prompt_path):
    prompt_speech_16k = load_wav(audio_prompt_path, 16000)
    # text_list = preprocess(text)
    text_list = [text]
    for i in text_list:
        output = cosyvoice.inference_cross_lingual(text, prompt_speech_16k)
        yield (22050, output['tts_speech'].numpy().flatten())


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>FunAudioLLM——Voice Translation👾</center>""")

    chatbot = gr.Chatbot(label='FunAudioLLM')
    with gr.Row():
        audio_input = gr.Audio(sources="microphone", label="Audio Input")
        audio_output = gr.Audio(label="Audio Output", autoplay=True, streaming=True)
        clear_button = gr.Button("Clear")

    audio_input.stop_recording(model_chat, inputs=[audio_input, chatbot], outputs=[chatbot, audio_output, audio_input])
    clear_button.click(clear_session, outputs=[chatbot, audio_output, audio_input])


if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(server_name='0.0.0.0', server_port=60002, ssl_certfile="../cert.pem", ssl_keyfile="../key.pem",
                inbrowser=True, ssl_verify=False)
