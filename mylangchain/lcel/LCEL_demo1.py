from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory

model = ChatOpenAI(
    api_key="sk-3d0b712661134d72991a4166262cbcea",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)


# load_dotenv()

prompt = ChatPromptTemplate.from_template("请将一个关于 {topic} 的笑话")

# model = SiliconflowFactory.get_default_model()
output_parser = StrOutputParser()

# 这里就是 LCEL
chain = prompt | model | output_parser

print(chain.invoke({"topic": "杰瑞"}))
