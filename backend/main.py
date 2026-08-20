from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field


load_dotenv()


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("data/sales.csv")


# ==================================================
# STRUCTURED ANALYSIS REQUEST
# ==================================================

class DataAnalysisRequest(BaseModel):

    operation: str = Field(
        description=(
            "The analysis operation to perform. "
            "Allowed operations are: "
            "average, sum, max, min, group_sum, "
            "group_max, top_n, filter_sum, compare."
        )
    )

    column: str | None = Field(
        default=None,
        description=(
            "The numeric column to analyze, "
            "such as Sales or Quantity."
        )
    )

    group_by: str | None = Field(
        default=None,
        description=(
            "The column to group by, "
            "such as City or Category."
        )
    )

    n: int | None = Field(
        default=None,
        description=(
            "Number of results for a top_n operation."
        )
    )

    filter_column: str | None = Field(
        default=None,
        description=(
            "The column to filter by, "
            "such as City or Category."
        )
    )

    filter_value: str | None = Field(
        default=None,
        description=(
            "The first value to filter for."
        )
    )

    compare_value: str | None = Field(
        default=None,
        description=(
            "The second value to compare against."
        )
    )


# ==================================================
# PANDAS ANALYSIS ENGINE
# ==================================================

def perform_analysis(request: DataAnalysisRequest) -> str:

    operation = request.operation.lower().strip()
    column = request.column
    group_by = request.group_by
    n = request.n
    filter_column = request.filter_column
    filter_value = request.filter_value
    compare_value = request.compare_value

    # --------------------------------------------------
    # Validate columns
    # --------------------------------------------------

    if column and column not in df.columns:

        return (
            f"Column '{column}' does not exist. "
            f"Available columns are: {', '.join(df.columns)}"
        )

    if group_by and group_by not in df.columns:

        return (
            f"Column '{group_by}' does not exist. "
            f"Available columns are: {', '.join(df.columns)}"
        )

    if filter_column and filter_column not in df.columns:

        return (
            f"Column '{filter_column}' does not exist. "
            f"Available columns are: {', '.join(df.columns)}"
        )

    # ==================================================
    # AVERAGE
    # ==================================================

    if operation == "average":

        if not column:
            return "Please specify which column to average."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."

        result = df[column].mean()

        return f"The average {column} is {result:.2f}."

    # ==================================================
    # SUM
    # ==================================================

    if operation == "sum":

        if not column:
            return "Please specify which column to sum."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."

        result = df[column].sum()

        return f"The total {column} is {result:.2f}."

    # ==================================================
    # MAXIMUM
    # ==================================================

    if operation == "max":

        if not column:
            return (
                "Please specify which column to find "
                "the maximum for."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."

        row = df.loc[df[column].idxmax()]

        if "Product" in df.columns:

            return (
                f"The highest {column} is "
                f"{row[column]:.2f}, "
                f"for the product {row['Product']}."
            )

        return (
            f"The highest {column} is "
            f"{row[column]:.2f}."
        )

    # ==================================================
    # MINIMUM
    # ==================================================

    if operation == "min":

        if not column:
            return (
                "Please specify which column to find "
                "the minimum for."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."

        row = df.loc[df[column].idxmin()]

        if "Product" in df.columns:

            return (
                f"The lowest {column} is "
                f"{row[column]:.2f}, "
                f"for the product {row['Product']}."
            )

        return (
            f"The lowest {column} is "
            f"{row[column]:.2f}."
        )

    # ==================================================
    # GROUP SUM
    # ==================================================

    if operation == "group_sum":

        if not column or not group_by:

            return (
                "Both a metric column and a "
                "group-by column are required."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):

            return f"Column '{column}' is not numeric."

        result = (
            df.groupby(group_by)[column]
            .sum()
            .sort_values(ascending=False)
        )

        return (
            f"Total {column} by {group_by}:\n\n"
            f"{result.to_string()}"
        )

    # ==================================================
    # GROUP MAX
    # ==================================================

    if operation == "group_max":

        if not column or not group_by:

            return (
                "Both a metric column and a "
                "group-by column are required."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):

            return f"Column '{column}' is not numeric."

        result = df.groupby(group_by)[column].sum()

        group = result.idxmax()
        value = result.max()

        return (
            f"{group} has the highest total "
            f"{column} with {value:.2f}."
        )

    # ==================================================
    # TOP N
    # ==================================================

    if operation == "top_n":

        if not column:
            return "Please specify which column to rank by."

        if not pd.api.types.is_numeric_dtype(df[column]):

            return f"Column '{column}' is not numeric."

        if n is None:
            n = 3

        if n <= 0:

            return (
                "The number of results must be "
                "greater than zero."
            )

        n = min(n, len(df))

        result = df.nlargest(n, column)

        return (
            f"Top {n} results by {column}:\n\n"
            f"{result.to_string(index=False)}"
        )

    # ==================================================
    # FILTER SUM
    # ==================================================

    if operation == "filter_sum":

        if not column:

            return "Please specify which column to sum."

        if not filter_column or not filter_value:

            return (
                "Both a filter column and "
                "filter value are required."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):

            return f"Column '{column}' is not numeric."

        filtered = df[
            df[filter_column]
            .astype(str)
            .str.lower()
            == str(filter_value).lower()
        ]

        if filtered.empty:

            return (
                f"No data was found where "
                f"{filter_column} is '{filter_value}'."
            )

        result = filtered[column].sum()

        return (
            f"The total {column} for "
            f"{filter_value} is {result:.2f}."
        )

    # ==================================================
    # COMPARE
    # ==================================================

    if operation == "compare":

        if not column:

            return (
                "Please specify which column "
                "to compare."
            )

        if not filter_column:

            return (
                "Please specify which column "
                "to compare values from."
            )

        if not filter_value or not compare_value:

            return (
                "Two values are required for "
                "comparison."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):

            return f"Column '{column}' is not numeric."

        # First value
        first_data = df[
            df[filter_column]
            .astype(str)
            .str.lower()
            == str(filter_value).lower()
        ]

        # Second value
        second_data = df[
            df[filter_column]
            .astype(str)
            .str.lower()
            == str(compare_value).lower()
        ]

        if first_data.empty:

            return (
                f"No data was found for "
                f"'{filter_value}'."
            )

        if second_data.empty:

            return (
                f"No data was found for "
                f"'{compare_value}'."
            )

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
                f"{filter_value} and {compare_value} "
                f"have the same total {column}: "
                f"{first_total:.2f}."
            )

        return (
            f"{filter_value}: {first_total:.2f}\n"
            f"{compare_value}: {second_total:.2f}\n\n"
            f"{winner} had higher total {column} "
            f"than {loser} by {difference:.2f}."
        )

    # ==================================================
    # UNKNOWN OPERATION
    # ==================================================

    return (
        f"I don't know how to perform the "
        f"operation '{request.operation}'."
    )


# ==================================================
# TOOLS
# ==================================================

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic addition calculations."""

    print("Calculator tool has been called.")

    return (
        f"The sum of {a} and {b} is {a + b}"
    )


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user."""

    print("Greeting tool has been called.")

    return (
        f"Hello {name}, I hope you are well today."
    )


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
    Use this tool for questions that require
    analyzing the sales dataset.

    Examples:

    "What is the average sales?"
    -> average / Sales

    "What are the total sales?"
    -> sum / Sales

    "Which city has the highest sales?"
    -> group_max / Sales / City

    "What are the total sales by category?"
    -> group_sum / Sales / Category

    "What are the top 3 products?"
    -> top_n / Sales / n=3

    "How much did Bangalore make?"
    -> filter_sum / Sales / City / Bangalore

    "Compare Bangalore and Mumbai sales."
    -> compare / Sales / City /
       Bangalore / Mumbai
    """

    print("Structured analysis tool has been called.")

    request = DataAnalysisRequest(
        operation=operation,
        column=column,
        group_by=group_by,
        n=n,
        filter_column=filter_column,
        filter_value=filter_value,
        compare_value=compare_value,
    )

    return perform_analysis(request)


# ==================================================
# MAIN CHATBOT
# ==================================================

def main():

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    tools = [
        calculator,
        say_hello,
        structured_analysis,
    ]

    system_prompt = f"""
    You are a helpful AI data assistant.

    You have access to a sales dataset.

    Dataset columns:
    {', '.join(df.columns)}

    Dataset rows:
    {len(df)}

    IMPORTANT RULES:

    1. When the user asks a question about
       the dataset, use structured_analysis.

    2. Never calculate dataset results yourself.

    3. Never use calculator for dataset questions.

    4. Choose the correct operation.

    5. Average sales:
       operation = average
       column = Sales

    6. Total sales:
       operation = sum
       column = Sales

    7. Highest sales by city/category:
       operation = group_max
       column = Sales
       group_by = City or Category

    8. Sales by category/city:
       operation = group_sum
       column = Sales
       group_by = Category or City

    9. Top products:
       operation = top_n
       column = Sales
       n = requested number

    10. Sales for a specific city/category:
        operation = filter_sum
        column = Sales
        filter_column = relevant column
        filter_value = requested value

    11. Comparing two cities, categories,
        products, or other values:

        operation = compare
        column = Sales
        filter_column = relevant column
        filter_value = first value
        compare_value = second value

        Example:

        "Compare Bangalore and Mumbai sales."

        operation = compare
        column = Sales
        filter_column = City
        filter_value = Bangalore
        compare_value = Mumbai

    12. Use conversation history to understand
        follow-up questions such as:

        "How much did it make?"

        "Which one was better?"

        "What about Mumbai?"

        "How much more did it make?"

    13. After the tool returns its result,
        explain it clearly.

    14. Do not repeatedly call the same tool
        for the same question.
    """

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    print("Welcome! I'm your AI assistant.")
    print("Type 'quit' to exit.")

    print(
        "You can ask me to perform calculations, "
        "greet someone, or analyze the dataset."
    )

    # ==================================================
    # CONVERSATION MEMORY
    # ==================================================

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

            # Add user message
            conversation.append(
                HumanMessage(
                    content=user_input
                )
            )

            response_messages = []

            # Send conversation to agent
            for chunk in agent.stream(
                {
                    "messages": conversation
                },
                config={
                    "recursion_limit": 10
                },
            ):

                if "model" in chunk:

                    messages = chunk[
                        "model"
                    ]["messages"]

                    for message in messages:

                        if message not in response_messages:

                            response_messages.append(
                                message
                            )

                        if message.type == "ai":

                            if isinstance(
                                message.content,
                                list
                            ):

                                for item in message.content:

                                    if (
                                        isinstance(
                                            item,
                                            dict
                                        )
                                        and item.get(
                                            "type"
                                        ) == "text"
                                    ):

                                        print(
                                            item.get(
                                                "text",
                                                ""
                                            ),
                                            end=""
                                        )

                            elif message.content:

                                print(
                                    message.content,
                                    end=""
                                )

            # Save response messages
            conversation.extend(
                response_messages
            )

            print()

        except Exception as e:

            print(
                f"\nSomething went wrong: {e}"
            )


if __name__ == "__main__":
    main()