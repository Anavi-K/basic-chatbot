import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from main import (
    create_chatbot,
    get_dataset_for_session,
    set_dataset_for_session,
    current_session_id_var,
    default_df,
)

load_dotenv()


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)
CORS(app)


# ==================================================
# MULTI-SESSION STORES
# ==================================================

conversations = {}          # session_id -> list of messages
session_dataset_info = {}   # session_id -> dict with metadata


def get_dataset_metadata(session_id: str) -> dict:
    if session_id in session_dataset_info:
        return session_dataset_info[session_id]

    target_df = get_dataset_for_session(session_id)
    return {
        "filename": "sales.csv (default)",
        "columns": list(target_df.columns),
        "rows": len(target_df),
        "is_default": True,
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Chatbot backend is running",
    })


# ==================================================
# UPLOAD DATASET ENDPOINT
# ==================================================

@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id", "default")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        contents = file.read()
        uploaded_df = pd.read_csv(io.BytesIO(contents))

        if uploaded_df.empty:
            return jsonify({"error": "The uploaded CSV file is empty"}), 400

        set_dataset_for_session(session_id, uploaded_df)

        metadata = {
            "filename": file.filename,
            "columns": list(uploaded_df.columns),
            "rows": len(uploaded_df),
            "is_default": False,
        }
        session_dataset_info[session_id] = metadata

        return jsonify({
            "status": "ok",
            "message": f"Successfully uploaded '{file.filename}' with {len(uploaded_df)} rows.",
            "dataset": metadata,
        })

    except Exception as e:
        print("Upload Error:", e)
        return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 500


# ==================================================
# GET DATASET METADATA ENDPOINT
# ==================================================

@app.route("/dataset", methods=["GET"])
def get_dataset_info():
    session_id = request.args.get("session_id", "default")
    return jsonify({
        "status": "ok",
        "dataset": get_dataset_metadata(session_id)
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
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({
            "error": "Message is required"
        }), 400

    token = current_session_id_var.set(session_id)
    try:
        agent = create_chatbot(session_id)

        history = conversations.get(session_id, [])
        history.append(HumanMessage(content=user_message))

        response = agent.invoke({
            "messages": history
        })

        messages = response["messages"]
        conversations[session_id] = messages

        for message in reversed(messages):
            if message.type == "ai":
                content = message.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = "".join(text_parts)

                return jsonify({
                    "response": content,
                    "session_id": session_id,
                    "dataset": get_dataset_metadata(session_id)
                })

        return jsonify({
            "response": "I couldn't generate a response.",
            "session_id": session_id
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "error": str(e)
        }), 500
    finally:
        current_session_id_var.reset(token)


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )