import ast
from contextvars import ContextVar
import math
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_manager import rag_instance


# ==================================================
# CONFIGURATION & DATASET MANAGEMENT
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
default_df = pd.read_csv(BASE_DIR / "data" / "sales.csv")

# Context variable to track active session ID per thread/request
current_session_id_var: ContextVar[str] = ContextVar("current_session_id_var", default="default")

# Multi-session dataset store: session_id -> pd.DataFrame
session_datasets = {}


def get_dataset_for_session(session_id: str = "default") -> pd.DataFrame:
    return session_datasets.get(session_id, default_df)


def set_dataset_for_session(session_id: str, dataframe: pd.DataFrame):
    session_datasets[session_id] = dataframe


# Backward compatibility export
df = default_df


# ==================================================
# STRUCTURED ANALYSIS REQUEST
# ==================================================

class DataAnalysisRequest(BaseModel):
    operation: str = Field(
        description=(
            "The analysis operation to perform. "
            "Allowed operations are: average, sum, max, min, "
            "group_sum, group_max, top_n, filter_sum, compare."
        )
    )

    column: str | None = Field(
        default=None,
        description="The numeric column to analyze.",
    )

    group_by: str | None = Field(
        default=None,
        description="The column to group by.",
    )

    n: int | None = Field(
        default=None,
        description="Number of results for a top_n operation.",
    )

    filter_column: str | None = Field(
        default=None,
        description="The column to filter by.",
    )

    filter_value: str | None = Field(
        default=None,
        description="The first value to filter for.",
    )

    compare_value: str | None = Field(
        default=None,
        description="The second value to compare against.",
    )


# ==================================================
# PANDAS ANALYSIS ENGINE
# ==================================================

def perform_analysis(request: DataAnalysisRequest, target_df: pd.DataFrame | None = None) -> str:

    if target_df is None or target_df.empty:
        target_df = default_df

    operation = request.operation.lower().strip()
    column = request.column
    group_by = request.group_by
    n = request.n
    filter_column = request.filter_column
    filter_value = request.filter_value
    compare_value = request.compare_value

    if column and column not in target_df.columns:
        return (
            f"Column '{column}' does not exist in the active dataset. "
            f"Available columns are: {', '.join(target_df.columns)}"
        )

    if group_by and group_by not in target_df.columns:
        return (
            f"Column '{group_by}' does not exist in the active dataset. "
            f"Available columns are: {', '.join(target_df.columns)}"
        )

    if filter_column and filter_column not in target_df.columns:
        return (
            f"Column '{filter_column}' does not exist in the active dataset. "
            f"Available columns are: {', '.join(target_df.columns)}"
        )

    if operation == "average":
        if not column:
            return "Please specify which column to average."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        result = target_df[column].mean()
        return f"The average {column} is {result:.2f}."

    if operation == "sum":
        if not column:
            return "Please specify which column to sum."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        result = target_df[column].sum()
        return f"The total {column} is {result:.2f}."

    if operation == "max":
        if not column:
            return "Please specify which column to find the maximum for."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        row = target_df.loc[target_df[column].idxmax()]

        label_cols = [c for c in target_df.columns if c != column and not pd.api.types.is_numeric_dtype(target_df[c])]
        if label_cols:
            primary_label = label_cols[0]
            return (
                f"The highest {column} is {row[column]:.2f}, "
                f"for ({primary_label}: {row[primary_label]})."
            )
        return f"The highest {column} is {row[column]:.2f}."

    if operation == "min":
        if not column:
            return "Please specify which column to find the minimum for."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        row = target_df.loc[target_df[column].idxmin()]

        label_cols = [c for c in target_df.columns if c != column and not pd.api.types.is_numeric_dtype(target_df[c])]
        if label_cols:
            primary_label = label_cols[0]
            return (
                f"The lowest {column} is {row[column]:.2f}, "
                f"for ({primary_label}: {row[primary_label]})."
            )
        return f"The lowest {column} is {row[column]:.2f}."

    if operation == "group_sum":
        if not column or not group_by:
            return "Both a metric column and a group-by column are required."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        result = (
            target_df.groupby(group_by)[column]
            .sum()
            .sort_values(ascending=False)
        )
        return f"Total {column} by {group_by}:\n\n{result.to_string()}"

    if operation == "group_max":
        if not column or not group_by:
            return "Both a metric column and a group-by column are required."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        result = target_df.groupby(group_by)[column].sum()
        group = result.idxmax()
        value = result.max()
        return f"'{group}' has the highest total {column} with {value:.2f}."

    if operation == "top_n":
        if not column:
            return "Please specify which column to rank by."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        if n is None:
            n = 3
        if n <= 0:
            return "The number of results must be greater than zero."
        n = min(n, len(target_df))
        result = target_df.nlargest(n, column)
        return f"Top {n} results by {column}:\n\n{result.to_string(index=False)}"

    if operation == "filter_sum":
        if not column:
            return "Please specify which column to sum."
        if not filter_column or not filter_value:
            return "Both a filter column and filter value are required."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."
        filtered = target_df[
            target_df[filter_column]
            .astype(str)
            .str.lower()
            == str(filter_value).lower()
        ]
        if filtered.empty:
            return f"No data was found where {filter_column} is '{filter_value}'."
        result = filtered[column].sum()
        return f"The total {column} for '{filter_value}' is {result:.2f}."

    if operation == "compare":
        if not column:
            return "Please specify which column to compare."
        if not filter_column:
            return "Please specify which column to compare values from."
        if not filter_value or not compare_value:
            return "Two values are required for comparison."
        if not pd.api.types.is_numeric_dtype(target_df[column]):
            return f"Column '{column}' is not numeric."

        first_data = target_df[
            target_df[filter_column]
            .astype(str)
            .str.lower()
            == str(filter_value).lower()
        ]
        second_data = target_df[
            target_df[filter_column]
            .astype(str)
            .str.lower()
            == str(compare_value).lower()
        ]

        if first_data.empty:
            return f"No data was found for '{filter_value}' in {filter_column}."
        if second_data.empty:
            return f"No data was found for '{compare_value}' in {filter_column}."

        first_total = first_data[column].sum()
        second_total = second_data[column].sum()
        difference = abs(first_total - second_total)

        if first_total > second_total:
            winner = filter_value
            loser = compare_value
        elif second_total > first_total:
            winner = compare_value
            loser = filter_value
        else:
            return (
                f"'{filter_value}' and '{compare_value}' "
                f"have the same total {column}: {first_total:.2f}."
            )

        return (
            f"{filter_value}: {first_total:.2f}\n"
            f"{compare_value}: {second_total:.2f}\n\n"
            f"'{winner}' had higher total {column} than '{loser}' by {difference:.2f}."
        )

    return f"I don't know how to perform the operation '{request.operation}'."


# ==================================================
# AGENT TOOLS
# ==================================================

@tool
def knowledge_base_search(query: str) -> str:
    """
    Retrieves grounded context from the company knowledge base (company overview, policies, product specs, FAQs).
    Includes similarity thresholding for hallucination guardrail protection.
    """
    print(f"Knowledge base search tool called for query: '{query}'")
    result = rag_instance.query_with_guardrail(query)

    if not result.get("success"):
        return result.get("message", "I don't have enough information in my knowledge base to answer that.")

    return (
        f"RETRIEVED KNOWLEDGE CONTEXT (Confidence Score: {result.get('best_score')}):\n\n"
        f"{result.get('context')}\n\n"
        f"INSTRUCTION: Answer the question accurately using ONLY the retrieved context above. "
        f"Do not invent unmentioned information."
    )


@tool
def advanced_calculator(expression: str) -> str:
    """
    Evaluates mathematical expressions including addition, subtraction,
    multiplication, division, percentages, powers/exponents, square roots, and parentheses.
    """
    print(f"Advanced calculator tool called with expression: {expression}")

    try:
        expr = expression.strip()

        expr = re.sub(
            r'(\d+(?:\.\d+)?)\%\s*of\s*(\d+(?:\.\d+)?)',
            r'(\1 / 100) * \2',
            expr,
            flags=re.IGNORECASE
        )
        expr = re.sub(r'(\d+(?:\.\d+)?)\%', r'(\1 / 100)', expr)
        expr = expr.replace('^', '**')

        allowed_funcs = {
            'sqrt': math.sqrt,
            'abs': abs,
            'round': round,
            'pow': math.pow,
            'ceil': math.ceil,
            'floor': math.floor,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }

        node = ast.parse(expr, mode='eval')

        def _eval(n):
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            elif isinstance(n, ast.Constant):
                return n.value
            elif isinstance(n, ast.Name):
                if n.id in allowed_funcs:
                    return allowed_funcs[n.id]
                raise ValueError(f"Unsupported symbol '{n.id}'")
            elif isinstance(n, ast.UnaryOp):
                op = n.op
                val = _eval(n.operand)
                if isinstance(op, ast.USub):
                    return -val
                elif isinstance(op, ast.UAdd):
                    return +val
                raise ValueError(f"Unsupported unary operator {type(op)}")
            elif isinstance(n, ast.BinOp):
                left = _eval(n.left)
                right = _eval(n.right)
                op = n.op
                if isinstance(op, ast.Add):
                    return left + right
                elif isinstance(op, ast.Sub):
                    return left - right
                elif isinstance(op, ast.Mult):
                    return left * right
                elif isinstance(op, ast.Div):
                    return left / right
                elif isinstance(op, ast.FloorDiv):
                    return left // right
                elif isinstance(op, ast.Mod):
                    return left % right
                elif isinstance(op, ast.Pow):
                    return left ** right
                raise ValueError(f"Unsupported binary operator {type(op)}")
            elif isinstance(n, ast.Call):
                func = _eval(n.func)
                args = [_eval(arg) for arg in n.args]
                return func(*args)
            else:
                raise ValueError(f"Unsupported AST node {type(n)}")

        res = _eval(node)
        if isinstance(res, float) and res.is_integer():
            res = int(res)

        return f"Calculation result for '{expression}': {res}"

    except Exception as e:
        return f"Error evaluating mathematical expression '{expression}': {e}"


@tool
def calculator(a: float, b: float) -> str:
    """Legacy calculator tool for addition."""
    return advanced_calculator.invoke({"expression": f"{a} + {b}"})


@tool
def text_utilities(action: str, text: str) -> str:
    """
    Performs specialized text transformation tasks on user-provided text.
    """
    print(f"Text utilities tool called with action '{action}'")

    action_clean = action.lower().strip()

    prompt_map = {
        "rewrite_professionally": "Rewrite the following text professionally for a formal business context:\n\n",
        "summarize": "Summarize the following text:\n\n",
        "make_concise": "Make the following text concise and to the point:\n\n",
        "explain_simply": "Explain the following paragraph in very simple terms:\n\n",
        "fix_grammar": "Fix all grammar and spelling errors:\n\n",
        "bullet_points": "Convert the following text into bullet points:\n\n",
    }

    instructions = prompt_map.get(
        action_clean,
        f"Perform '{action}' on the text:\n\n"
    )

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.3,
        )

        response = model.invoke(f"{instructions}{text}")

        content = response.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "".join(text_parts)

        return content.strip()

    except Exception as e:
        return f"Error executing text utility transformation: {e}"


@tool
def web_search(query: str) -> str:
    """
    Searches the web for real-time information or questions outside the dataset and knowledge base.
    """
    print(f"Web search tool called for query: {query}")

    try:
        results = list(DDGS().text(query, max_results=5))

        if not results:
            return f"No web search results found for query: '{query}'"

        formatted = [f"Web Search Results for '{query}':\n"]
        for i, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            snippet = item.get("body", "No Snippet")
            href = item.get("href", "")
            formatted.append(f"{i}. {title}\n   Snippet: {snippet}\n   Link: {href}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Failed to perform web search: {e}"


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user."""
    print("Greeting tool has been called.")
    return f"Hello {name}, I hope you are well today."


@tool
def structured_analysis(
    operation: str,
    column: str | None = None,
    group_by: str | None = None,
    n: int | None = None,
    filter_column: str | None = None,
    filter_value: str | None = None,
    compare_value: str | None = None,
) -> str:
    """
    Use this tool for questions that require analyzing a tabular dataset (uploaded CSV or default dataset).
    """
    print("Structured analysis tool has been called.")

    sid = current_session_id_var.get()
    target_df = get_dataset_for_session(sid)

    request = DataAnalysisRequest(
        operation=operation,
        column=column,
        group_by=group_by,
        n=n,
        filter_column=filter_column,
        filter_value=filter_value,
        compare_value=compare_value,
    )

    return perform_analysis(request, target_df)


def create_chatbot(session_id: str = "default"):
    target_df = get_dataset_for_session(session_id)

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    tools = [
        knowledge_base_search,
        structured_analysis,
        advanced_calculator,
        text_utilities,
        web_search,
        say_hello,
    ]

    cols_str = ", ".join(target_df.columns)

    system_prompt = f"""
    You are a powerful AI assistant equipped with specialized tools.

    YOUR TOOLS & WHEN TO USE THEM:

    1. `knowledge_base_search`: Use for questions about company history, locations, operating hours, shipping/return policies, product catalog specs, warranty, or FAQs.
       - Built-in Hallucination Guardrail: Returns "I don't have enough information in my knowledge base to answer that." if relevance is insufficient.

    2. `structured_analysis`: Use whenever the user asks a question about the active CSV dataset.
       Current Dataset Columns: {cols_str}
       Total Rows: {len(target_df)}

    3. `advanced_calculator`: Use for ANY math calculations, expressions, percentages, powers, or square roots requested by the user.

    4. `text_utilities`: Use when the user asks to edit, rewrite, summarize, explain, or format text/email/paragraphs.

    5. `web_search`: Use when the user asks for real-time external web news or questions NOT related to the knowledge base or active dataset.

    6. `say_hello`: Use to greet the user.

    IMPORTANT RULES:
    - Never guess company policies or knowledge base details; use `knowledge_base_search`.
    - If `knowledge_base_search` returns "I don't have enough information in my knowledge base to answer that.", repeat that exact guardrail statement to the user.
    - Never perform dataset queries manually; always use `structured_analysis`.
    - Respond naturally, clearly, and concisely.
    """

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


def main():
    agent = create_chatbot("default")

    print("Welcome! I'm your AI assistant.")
    print("Type 'quit' to exit.")

    conversation = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nAssistant: ", end="")

        try:
            conversation.append(HumanMessage(content=user_input))
            response_messages = []

            for chunk in agent.stream(
                {"messages": conversation},
                config={"recursion_limit": 10},
            ):
                if "model" not in chunk:
                    continue

                messages = chunk["model"]["messages"]
                for message in messages:
                    if message not in response_messages:
                        response_messages.append(message)

                    if message.type != "ai":
                        continue

                    if isinstance(message.content, list):
                        for item in message.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                print(item.get("text", ""), end="")
                    elif message.content:
                        print(message.content, end="")

            conversation.extend(response_messages)
            print()

        except Exception as e:
            print(f"\nSomething went wrong: {e}")


if __name__ == "__main__":
    main()