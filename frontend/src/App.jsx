import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  // Scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  const sendMessage = async (messageText = input) => {
    const message = messageText.trim();

    if (!message || loading) {
      return;
    }

    // Add user's message
    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: message,
      },
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Something went wrong."
        );
      }

      // Add assistant response
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.response,
        },
      ]);
    } catch (error) {
      console.error("Chat error:", error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I couldn't process that request right now. Please try again later.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Enter sends the message
  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  };

  const handleExampleClick = (question) => {
    sendMessage(question);
  };

  return (
    <div className="app">

      <div className="chat-container">

        {/* Header */}
        <header className="chat-header">

          <div className="header-title">

            <div className="logo">
              ↂ
            </div>

            <div>
              <h1>AI Assistant</h1>

              <p>
                Chat, calculate, and explore your data
              </p>
            </div>

          </div>

        </header>


        {/* Messages */}
        <main className="messages">

          {/* Welcome screen */}
          {messages.length === 0 && !loading && (
            <div className="welcome">

              <div className="welcome-icon">
                ↂ
              </div>

              <h2>
                What can I help you with?
              </h2>

              <p>
                Ask me a question, perform a calculation,
                or explore the sales dataset.
              </p>

              <div className="examples">

                <button
                  onClick={() =>
                    handleExampleClick(
                      "Which city has the highest sales?"
                    )
                  }
                >
                  <strong>
                    Analyze your data
                  </strong>

                  <span>
                    Which city has the highest sales?
                  </span>
                </button>


                <button
                  onClick={() =>
                    handleExampleClick(
                      "What are the top 3 products?"
                    )
                  }
                >
                  <strong>
                    Explore products
                  </strong>

                  <span>
                    What are the top 3 products?
                  </span>
                </button>


                <button
                  onClick={() =>
                    handleExampleClick(
                      "What is 125 multiplied by 48?"
                    )
                  }
                >
                  <strong>
                    Quick calculation
                  </strong>

                  <span>
                    What is 24 x 7?
                  </span>
                </button>

              </div>

            </div>
          )}


          {/* Chat messages */}
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role}`}
            >

              {message.role === "assistant" && (
                <div className="message-avatar">
                  ↂ
                </div>
              )}

              <div className="message-bubble">
                {message.content}
              </div>

            </div>
          ))}


          {/* Loading animation */}
          {loading && (
            <div className="message assistant">

              <div className="message-avatar">
                ↂ
              </div>

              <div className="message-bubble loading-bubble">

                <span></span>
                <span></span>
                <span></span>

              </div>

            </div>
          )}

          <div ref={messagesEndRef} />

        </main>


        {/* Input */}
        <div className="input-area">

          <div className="input-container">

            <input
              type="text"
              placeholder="Ask me anything..."
              value={input}
              onChange={(event) =>
                setInput(event.target.value)
              }
              onKeyDown={handleKeyDown}
              disabled={loading}
            />

            <button
              className="send-button"
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              aria-label="Send message"
            >
              ↑
            </button>

          </div>

          <p className="input-hint">
            AI can make mistakes. Check important information.
          </p>

        </div>

      </div>

    </div>
  );
}

export default App;