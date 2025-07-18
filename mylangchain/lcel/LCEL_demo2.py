from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory


# 输出结构
class SortEnum(str, Enum):
    mobile_data = 'mobile_data'
    price = 'price'


class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'


class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    mobile_data_lower: Optional[int] = Field(description="流量下限", default=None)
    mobile_data_upper: Optional[int] = Field(description="流量上限", default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序", default=None)
    ordering: Optional[OrderingEnum] = Field(
        description="升序或降序排列", default=None)


if __name__ == "__main__":
    load_dotenv()

    parser = PydanticOutputParser(pydantic_object=Semantics)
    prompt = PromptTemplate(
        template="你是一个语义解析器。你的任务是将用户的输入解析成JSON表示。不要回答用户的问题。\n"
                 "用户输入:{text}\n"
                 "{format_instructions}",
        # input_variables 表示输入变量，text是用户输入
        input_variables=["text"],
        # partial_variables 表示部分变量，format_instructions是解析器
        # get_format_instructions() 获取解析器的格式化指令，按照Semantics格式解析
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = ChatOpenAI(
        api_key="sk-3d0b712661134d72991a4166262cbcea",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # other params...
    )

    # LCEL 表达式
    # RunnablePassthrough() 表示将输入原封不动地传递给下一个组件,
    # 数据传递：它接收输入并原样输出，不进行任何处理
    runnable = (
            {"text": RunnablePassthrough()} | prompt | model | parser
    )

    # 直接运行
    ret = runnable.invoke({"text": "不超过100元的套餐哪个流量最大"})

    print(ret)
