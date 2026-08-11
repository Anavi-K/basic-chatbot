from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers."""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user."""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"


def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )

    tools = [calculator, say_hello]

    agent = create_agent(
        model=model,
        tools=tools,
    )

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="")

        for chunk in agent.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "model" in chunk:
                message = chunk["model"]["messages"][-1]

                if isinstance(message.content, list):
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            print(item.get("text", ""), end="")
                else:
                    print(message.content, end="")

        print()


if __name__ == "__main__":
    main()