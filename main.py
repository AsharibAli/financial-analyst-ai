import time  # Imports the time module for handling time-related tasks.
import openai  # Imports the OpenAI library for accessing OpenAI's API.
from dotenv import (
    load_dotenv,
    find_dotenv,
)  # Imports functions to handle environment variables.
import requests  # Imports the requests module for making HTTP requests.
import json  # Imports the json module for JSON manipulation.
import os  # Imports the os module to interact with the operating system.

# OpenAI specific imports for handling various types and structures.
from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.threads.run import Run

# Read local .env file to load environment variables (like API keys).
_: bool = load_dotenv(find_dotenv())  # read local .env file

# Retrieves the FMP API Key from the environment variables.
FMP_API_KEY: str | None = os.getenv("FMP_API_KEY")

# Initializes an OpenAI client.
client: openai.OpenAI = openai.OpenAI()


# Define functions to retrieve financial data using the Financial Modeling Prep API.
# Each function takes a stock ticker, a period (annual/quarterly), and a limit as arguments
# and returns the requested financial data in JSON format.


def get_income_statement(ticker: str, period: str, limit: int) -> str:
    # Function to get income statement data.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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
    # Function to get balance sheet data.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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
    # Function to get cash flow statement data.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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
    # Function to get key metrics for a company.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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
    # Function to get financial ratios.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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
    # Function to get financial growth data.
    # Constructs the request URL with parameters and sends a GET request.
    # Returns the response in JSON format.
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


# A dictionary mapping function names to their corresponding functions.
available_functions: dict = {
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow_statement": get_cash_flow_statement,
    "get_key_metrics": get_key_metrics,
    "get_financial_ratios": get_financial_ratios,
    "get_financial_growth": get_financial_growth,
}


# Defines a function to run the OpenAI assistant.
def run_assistant(user_message: str):
    # Creates an OpenAI assistant instance with specific instructions and tools.
    # Each tool corresponds to a financial data retrieval function.
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
    # Creates a new thread for handling the conversation.
    thread: Thread = client.beta.threads.create()

    # Adds the user's message to the thread.
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    # Starts running the assistant on the thread.
    run: Run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id
    )
    # Enters a loop to handle the assistant's responses and actions.
    while True:
        # Retrieves the current status of the run.
        runStatus = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

        # Handles cases where the assistant requires action (like calling a function).
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
                # Loops through required tool calls.
                for toolcall in toolCalls:
                    # Executes the corresponding function and captures the output.
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

                # Submits the tool outputs back to the assistant.
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )
        # Handles the case where the assistant's run is completed.
        elif runStatus.status == "completed":
            # Retrieves and returns the final messages from the assistant.

            messages: list[ThreadMessage] = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            for message in messages.data:
                message_content = message.content[0].text.value
                return message_content
            break  # Exit the loop after processing the completed run
        # Handles other statuses like 'failed', 'in_progress', or 'queued'.

        elif run.status == "failed":
            print("Run failed.")
            break

        elif run.status in ["in_progress", "queued"]:
            print(f"Run is {run.status}. Waiting...")
            time.sleep(5)  # Wait for 5 seconds before checking again

        else:
            print(f"Unexpected status: {run.status}")
            break


# The code above sets up an advanced AI-powered financial analysis tool that can respond to user queries with specific financial data. It integrates OpenAI's GPT-3.5 model for conversational capabilities and uses the Financial Modeling Prep API to fetch real-time financial data. The application is designed to be interactive and user-friendly, providing detailed financial insights in response to user inputs.
