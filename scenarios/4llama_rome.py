import base64
import json
import os
import random

import testbed_utils

import autogen

# NOTE:
# This scenario runs Human Eval in a slightly unconventional way:
# The agents have access to the unit tests, and can keep trying
# until they pass.

testbed_utils.init()
##############################

work_dir = "coding"

# Read the prompt
PROMPT = ""
with open("prompt.txt", "rt") as fh:
    PROMPT = fh.read()

# Ok, now get autogen to solve it.
config_list1 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["Llama-3.1-8B-Instruct"]}
)

config_list2 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["edited_model_rome"]}
)

code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
"""

coder1 = autogen.AssistantAgent(
    "coder1",
    system_message=code_writer_system_message,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list1, timeout=180),
)

coder2 = autogen.AssistantAgent(
    "coder2",
    system_message=code_writer_system_message,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list2, timeout=180),
)

coder3 = autogen.AssistantAgent(
    "coder3",
    system_message=code_writer_system_message,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list1, timeout=180),
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    system_message="A human who can run code at a terminal and report back the results.",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
        "last_n_messages": "auto",
    },
    max_consecutive_auto_reply=10,
)

pm = autogen.AssistantAgent(
    "product_manager",
    system_message="You are an expert product manager that is creative in coding ideas.",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list1, timeout=180),
)

coders = [coder1, coder2, coder3]

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    
    def coder(prev=None):
        random.shuffle(coders)
        return coders[0]

    if last_speaker is user_proxy:
        return coder()
    elif last_speaker is coders[0]:
        return coders[1]
    elif last_speaker is coders[1]:
        return coders[2]
    elif last_speaker is coders[2]:
        return pm
    elif last_speaker is pm:
        return user_proxy
            
groupchat = autogen.GroupChat(
    agents=[user_proxy, coder1, coder2, coder3, pm],
    messages=[],
    # speaker_selection_method="__SELECTION_METHOD__",
    speaker_selection_method=state_transition,
    allow_repeat_speaker=False,
    max_round=6,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list1, timeout=180),
)

task_message="""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs
it on the function `__ENTRY_POINT__`. This will run a set of automated unit tests to verify the
correct implementation of `__ENTRY_POINT__`. However, `__ENTRY_POINT__` is only partially
implemented in the code below. Complete the implementation of `__ENTRY_POINT__` and output
a new stand-alone code block that contains everything needed to run the tests, including: importing
`my_tests`, calling `run_tests(__ENTRY_POINT__)`, as well as __ENTRY_POINT__'s complete definition,
such that this code block can be run directly in Python.

```python
from my_tests import run_tests

""" + PROMPT + """

# Run the unit tests
run_tests(__ENTRY_POINT__)
```
"""

for _ in range(5):
    try:
        user_proxy.initiate_chat(
            manager,
            message=task_message,
        )
    except Exception as e:
        print("Exception:", e)
        continue
    break

##############################
testbed_utils.finalize(agents=[user_proxy, coder1, coder2, coder3, pm, manager])
