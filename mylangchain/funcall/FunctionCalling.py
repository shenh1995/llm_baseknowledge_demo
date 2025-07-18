from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from ..llms.siliconflow.Siliconflow import SiliconflowFactory


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


if __name__ == "__main__":
    load_dotenv()

    llm = SiliconflowFactory.get_default_model()
    
    if llm is None:
        print("Failed to get default model")
        exit(1)

    llm_with_tools = llm.bind_tools([add, multiply])

    query = "What is 3 * 12? Also"
    messages: list[BaseMessage] = [HumanMessage(query)]

    output = llm_with_tools.invoke(messages)
    print(output)

    # 回传 Function Call 的结果
    messages.append(output)

    available_tools = {"add": add, "multiply": multiply}

    # 检查output是否有tool_calls属性
    if isinstance(output, AIMessage) and hasattr(output, 'tool_calls') and output.tool_calls:
        for tool_call in output.tool_calls:
            selected_tool = available_tools[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg)


