#!/usr/bin/env python3
"""
运行FunctionCalling模块的脚本
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行FunctionCalling模块
from mylangchain.funcall.FunctionCalling import *

if __name__ == "__main__":
    # 直接运行main部分的代码
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