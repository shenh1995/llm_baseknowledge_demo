import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class SiliconflowFactory:
    model_params = {
        "temperature": 0,  # 适用于需要确定性回答的场景，如程序代码生成、自动化文档撰写、数据分析等
        "seed": 42,  # 输出将完全可复现，每次运行都生成相同的结果
    }

    @classmethod
    def get_model(cls, model_name: str):
        if model_name == "qianwen":
            return ChatOpenAI(
                api_key="sk-3d0b712661134d72991a4166262cbcea",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                **cls.model_params,
                # other params...
                )
        elif model_name == "qwen-plus":
            return ChatOpenAI(
                model="deepseek-ai/DeepSeek-V3",  # 模型名称
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
                openai_api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
                **cls.model_params,
            )

    @classmethod
    def get_default_model(cls):
        return cls.get_model(model_name="qianwen")


if __name__ == "__main__":
    load_dotenv()
    llm = SiliconflowFactory.get_default_model()
    response = llm.invoke("你是谁？")
    print(response.content)
