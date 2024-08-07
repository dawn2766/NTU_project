import gradio as gr
import json
from http import HTTPStatus
from dashscope import Application

with open("./api.json", "r", encoding='utf-8') as f:
    data = json.load(f)
    app_id_VisionVoice = data["bailian"]["app_id_NTU"]
    api_key_bailian = data["bailian"]["api_key"]

# 保存用户聊天历史的session
session_id = ""


def predict(image, user_talk):
    """
    生成器函数，用于大模型流式输出
    """
    global session_id
    response_all = ""
    text_before = ""

    print(type(image))

    if not session_id:
        responses = Application.call(app_id=app_id_VisionVoice,
                                     prompt='我上传了一张棕熊在嘻戏的图片，总共有四只棕熊，水流湍急。请你记住这些图片信息并进行适当拓展，以回答我的问题。我的提问：'+user_talk,
                                     api_key=api_key_bailian,
                                     stream=True
                                     )
    else:
        responses = Application.call(app_id=app_id_VisionVoice,
                                     prompt='我上传了一张棕熊在嘻戏的图片，总共有四只棕熊，水流湍急。请你记住这些图片信息并进行适当拓展，以回答我的问题。我的提问：'+user_talk,
                                     api_key=api_key_bailian,
                                     stream=True,
                                     #  history=multi_talks
                                     session_id=session_id
                                     )

    # 大模型流式输出
    for response in responses:
        if not session_id:
            session_id = response['output']['session_id']
        if response.status_code != HTTPStatus.OK:
            yield '\nrequest_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message)
        else:
            text = response["output"]["text"][len(text_before):]
            response_all += text
            text_before = response_all
            yield response_all


# 创建 Gradio 接口，包括图像和文本输入
iface = gr.Interface(fn=predict, inputs=["image", "text"],
                     outputs="text", title="NTU Project Web UI")
iface.launch()
