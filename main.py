import time
import openai
from dotenv import load_dotenv, find_dotenv
import requests
import json
import os


from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.threads.run import Run


_: bool = load_dotenv(find_dotenv())  # read local .env file

FMP_API_KEY: str | None = os.getenv("FMP_API_KEY")
client: openai.OpenAI = openai.OpenAI()


# Define financial statement functions
def get_income_statement(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves income statement data for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol.
        period (str): The period (e.g., 'annual' or 'quarterly').
        limit (int): The maximum number of records to retrieve.

    Returns:
        str: JSON string containing income statement data.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


def get_balance_sheet(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves the balance sheet statement for a given ticker.

    Args:
        ticker (str): The ticker symbol of the company.
        period (str): The period of the balance sheet statement (e.g., 'annual', 'quarter').
        limit (int): The number of periods to retrieve.

    Returns:
        str: The balance sheet statement in JSON format.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


def get_cash_flow_statement(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves the cash flow statement for a given ticker.

    Args:
        ticker (str): The ticker symbol of the company.
        period (str): The period for which the cash flow statement is requested (e.g., annual, quarterly).
        limit (int): The number of periods to retrieve.

    Returns:
        str: The cash flow statement in JSON format.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


def get_key_metrics(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves key metrics for a given ticker.

    Args:
        ticker (str): The ticker symbol of the company.
        period (str): The period for which key metrics are retrieved.
        limit (int): The maximum number of key metrics to retrieve.

    Returns:
        str: A JSON string containing the key metrics data.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


def get_financial_ratios(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves financial ratios for a given ticker.

    Args:
        ticker (str): The ticker symbol of the company.
        period (str): The period for which the ratios are requested (e.g., annual, quarterly).
        limit (int): The maximum number of ratios to retrieve.

    Returns:
        str: A JSON string containing the financial ratios.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


def get_financial_growth(ticker: str, period: str, limit: int) -> str:
    """
    Retrieves the cash flow statement growth data for a given ticker.

    Args:
        ticker (str): The ticker symbol of the company.
        period (str): The time period for the data (e.g., 'annual', 'quarter').
        limit (int): The number of records to retrieve.

    Returns:
        str: The JSON string representation of the response data.
    """
    url: str = f"https://financialmodelingprep.com/api/v3/cash-flow-statement-growth/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response: requests.Response = requests.get(url)
    return json.dumps(response.json())


# Map available functions
available_functions: dict = {
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow_statement": get_cash_flow_statement,
    "get_key_metrics": get_key_metrics,
    "get_financial_ratios": get_financial_ratios,
    "get_financial_growth": get_financial_growth,
}


# Define the assistant function
def run_assistant(user_message: str):
    # Creating an assistant with specific instructions and tools
    assistant: Assistant = client.beta.assistants.create(
        instructions="Act as a financial analyst by accessing detailed financial data through the Financial Modeling Prep API. Your capabilities include analyzing key metrics, comprehensive financial statements, vital financial ratios, and tracking financial growth trends. ",
        model="gpt-3.5-turbo-1106",
        tools=[
            # The first tool is a function that retrieves income statement data
            {
                "type": "function",
                "function": {
                    "name": "get_income_statement",
                    "description": "Retrieves income statement data for a given stock ticker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["ticker"],
                    },
                },
            },
            # The second tool is a function that retrieves balance sheet data
            {
                "type": "function",
                "function": {
                    "name": "get_balance_sheet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
            # The third tool is a function that retrieves cash flow statement data
            {
                "type": "function",
                "function": {
                    "name": "get_cash_flow_statement",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
            # The fourth tool is a function that retrieves key metrics
            {
                "type": "function",
                "function": {
                    "name": "get_key_metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
            # The fifth tool is a function that retrieves financial ratios
            {
                "type": "function",
                "function": {
                    "name": "get_financial_ratios",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
            # The sixth tool is a function that retrieves financial growth data
            {
                "type": "function",
                "function": {
                    "name": "get_financial_growth",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "period": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
        ],
    )
    # Creating a new thread
    thread: Thread = client.beta.threads.create()

    # First Request Adding a user message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    # Running the assistant on the created thread
    run: Run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id
    )

    while True:
        runStatus = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

        # This means run is making a function call
        if (
            runStatus.status == "requires_action"
            and runStatus.required_action is not None
        ):
            if (
                runStatus.required_action.submit_tool_outputs
                and runStatus.required_action.submit_tool_outputs.tool_calls
            ):
                toolCalls = runStatus.required_action.submit_tool_outputs.tool_calls
                tool_outputs: list[ToolOutput] = []
                for toolcall in toolCalls:
                    function_name = toolcall.function.name
                    function_args = json.loads(toolcall.function.arguments)

                    if function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        response = function_to_call(**function_args)
                        tool_outputs.append(
                            {
                                "tool_call_id": toolcall.id,
                                "output": response,
                            }
                        )

                # Submit tool outputs and update the run
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )

        elif runStatus.status == "completed":
            messages: list[ThreadMessage] = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            for message in messages.data:
                message_content = message.content[0].text.value
                return message_content
            break  # Exit the loop after processing the completed run
        elif run.status == "failed":
            print("Run failed.")
            break

        elif run.status in ["in_progress", "queued"]:
            print(f"Run is {run.status}. Waiting...")
            time.sleep(5)  # Wait for 5 seconds before checking again

        else:
            print(f"Unexpected status: {run.status}")
            break
