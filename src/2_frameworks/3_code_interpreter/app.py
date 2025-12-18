"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""

from pathlib import Path

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    CodeInterpreter,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()

CODE_INTERPRETER_INSTRUCTIONS = """\
The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.
"""

AGENT_LLM_NAME = "gemini-2.5-flash"
async_openai_client = AsyncOpenAI()
code_interpreter = CodeInterpreter(
    local_files=[
        Path("sandbox_content/"),
        Path("tests/tool_tests/example_files/data_b.csv"),
    ]
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    main_agent = agents.Agent(
        name="Data Analysis Agent",
        instructions=CODE_INTERPRETER_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="OAI Agent SDK ReAct + LangFuse Code Interpreter + Financial Transaction Data",
    type="messages",
    examples=[
        "Describe the data in data_b.csv",
        "In the data_b.csv file load as df and count the number of transaction in each state using the merchant state as the location of the transaction",
        "In data_b.csv, Create a table for current_age range 50-55, 60-65 etc. and count the occurrences of target = Yes for each range, print it out",
        "Use the file data_b.csv. In this file load contents to Data frame. Convert column target into binary 0/1, convert any columns with object representing money to float and loop through amount, per_capita_income, yearly_income, total_debt, credit_limit column and perform student t test to compare the means between target = 1 and target = 0 labels. what are the results of the test",
        "In the data_b.csv file,  use one hot encoding for binary data, calculate the statistical fit of a gaussian for target = no and for target = yes on the transaction amount. Fit and print the amplitude, x offset and sigma of each gaussian with uncertainties.",
        "Load the data_b.csv file into a data frame. Perform one hot encoding and clean all money columns to double and yes no to binary build a Random Forest Classifier ML model to predict target, do not use clinet_id, card_id, or id as features. train the model on 80 perecent of data and on the other 20 do a validation on the tartget with f1 and the recall on each target type",
        "Load the data_b.csv file into a data frame. Perform one hot encoding and clean all money columns to double and yes no to binary build  a gradient boosting classifier to predict target, do not use clinet_id, card_id, or id as features. train the model on 80 precent of data and on the other 20 do a validation on the target with f1 and the recall on each target type",
    ],
)

example = '''' 

'''
if __name__ == "__main__":
    demo.launch(share=True)
