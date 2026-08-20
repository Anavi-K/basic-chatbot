from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv

from main import (
    calculator,
    say_hello,
    structured_analysis,
)

load_dotenv()


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)
CORS(app)


# ==================================================
# AI MODEL
# ==================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
)


# ==================================================
# AGENT TOOLS
# ==================================================

tools = [
    calculator,
    say_hello,
    structured_analysis,
]


# ==================================================
# SYSTEM PROMPT
# ==================================================

system_prompt = """
You are a helpful AI data assistant.

You have access to a sales dataset.

IMPORTANT RULES:

1. When the user asks a question about the
   dataset, use the structured_analysis tool.

2. Never calculate dataset results yourself.

3. Never use calculator for dataset questions.

4. Use the correct structured analysis operation.

5. For average sales:
   operation = average
   column = Sales

6. For total sales:
   operation = sum
   column = Sales

7. For highest sales by city/category:
   operation = group_max
   column = Sales
   group_by = City or Category

8. For sales by category/city:
   operation = group_sum
   column = Sales
   group_by = Category or City

9. For top products:
   operation = top_n
   column = Sales
   n = requested number

10. Use conversation history to understand
    follow-up questions.

11. If the user says things like:
    "how much did it make?"
    "what about that city?"
    "how many were sold?"
    use the previous conversation
    to understand what they are referring to.

12. After the tool returns its result,
    explain the result clearly.

13. Do not repeatedly call the same tool
    for the same question.
"""


# ==================================================
# CREATE AGENT
# ==================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)


# ==================================================
# CONVERSATION MEMORY
# ==================================================

conversation_history = []


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "message": "Chatbot backend is running"
    })


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body is required"
        }), 400

    user_message = data.get("message")

    if not user_message:
        return jsonify({
            "error": "Message is required"
        }), 400

    try:

        # ------------------------------------------
        # Add user message to conversation
        # ------------------------------------------

        conversation_history.append(
            HumanMessage(
                content=user_message
            )
        )

        # ------------------------------------------
        # Send entire conversation to agent
        # ------------------------------------------

        response = agent.invoke({
            "messages": conversation_history
        })

        messages = response["messages"]

        # ------------------------------------------
        # Update conversation history
        # ------------------------------------------

        conversation_history.clear()
        conversation_history.extend(messages)

        # ------------------------------------------
        # Find final AI response
        # ------------------------------------------

        for message in reversed(messages):

            if message.type == "ai":

                content = message.content

                # Gemini sometimes returns content
                # as a list of dictionaries

                if isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                        ):
                            text_parts.append(
                                item.get("text", "")
                            )

                    content = "".join(text_parts)

                return jsonify({
                    "response": content
                })

        return jsonify({
            "response": "I couldn't generate a response."
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )