from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

#注意：函数注释非常重要，它告诉LLM调用它能解决什么问题，
#get_word_length函数就是告诉LLM调用他可以计算单词长度。
@tool
def get_word_length(word: str) -> int:
    """返回单词的长度。"""
    return len(word)


tools = [get_word_length]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个非常强大的助手，但不了解当前事件。",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm =  SiliconflowFactory.get_default_model()
# 将工具绑定到模型中
llm_with_tools = llm.bind_tools(tools)


# 定义一个chain,
# step1: 为prompt模板准备参数，从agent调用输入中提取input和intermediate_steps两个参数，
#         input由用户输入，intermediate_steps由agent生成
# step2: 根据step1的参数，格式化prompt template
# step3: 调用模型
# step4: 处理模型输出
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# verbose = True代表在控制台打印详细的日志
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 调用agent
list(agent_executor.stream({"input": "eudca这个单词有几个字母"}))
