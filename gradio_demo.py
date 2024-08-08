import gradio as gr
import json
from http import HTTPStatus
from dashscope import Application
import torch
import torch.nn as nn
import torch.utils
from torchvision import transforms


labels = ["The old main gate of Nanyang Technological University", "Nanyang Technological University Memorial", "The interior of Xiao Long Bao", "The exterior of Xiao Long Bao",
          "Nanyang Business School", "The Computing and Data Science (CCDS) building", "The Chinese Heritage Centre", "The school plaque", "Yunnan Garden", "A corner of the teaching building"]

net = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 54 * 54, 240), nn.ReLU(),
    nn.Linear(240, 84), nn.ReLU(),
    nn.Linear(84, 10))

# 指定.pth文件的路径
file_path = "LeNet.pth"

# 加载.pth文件
model_weights = torch.load(file_path,map_location=torch.device('cpu'))

# 加载后可以将这些权重应用于你的模型
# 假设你有一个叫做 model 的模型
net.load_state_dict(model_weights)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像，单通道
    # 将张量数据归一化到 [-1, 1] 的范围
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def classify_img(image):
    image = transform(torch.tensor(image).permute(2, 0, 1).float())
    image = image.unsqueeze(0)
    y_hat = net(image)
    max_indices = torch.argmax(y_hat, dim=1)
    pridict_label = labels[max_indices[0].item()]
    return pridict_label


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

    image_label = classify_img(image)
    # print(image_label)
    image_label = "Xiao Long Bao"
    if not session_id:
        responses = Application.call(app_id=app_id_VisionVoice,
                                     prompt=f"I uploaded an image of '{image_label}' at Nanyang Technological University in Singapore, please tell me the context of this image first. Then use the information of '{image_label}' to answer my question. My question is:"+user_talk,
                                     api_key=api_key_bailian,
                                     stream=True
                                     )
    else:
        responses = Application.call(app_id=app_id_VisionVoice,
                                     prompt=f"I uploaded an image of the '{image_label}' at Nanyang Technological University in Singapore, please tell me the context of this image first. Then use the information of '{image_label}' to answer my question. My question is:"+user_talk,
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
