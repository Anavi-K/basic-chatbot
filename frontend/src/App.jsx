import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "https://chatbot-xven.onrender.com";

const STORAGE_KEY = "chatbot_conversations_v1";

function App() {
  const [conversations, setConversations] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.error("Failed to load conversation history:", e);
    }
    const newId = "session_" + Date.now();
    return [
      {
        id: newId,
        title: "New Chat",
        updatedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        messages: [],
        datasetInfo: null,
      },
    ];
  });

  const [currentSessionId, setCurrentSessionId] = useState(() => {
    return conversations[0]?.id || "session_" + Date.now();
  });

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.error("Failed to save conversation history:", e);
    }
  }, [conversations]);

  // Current active conversation object
  const currentConversation =
    conversations.find((c) => c.id === currentSessionId) || conversations[0];

  const messages = currentConversation?.messages || [];
  const datasetInfo = currentConversation?.datasetInfo || null;

  // Scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  const createNewChat = () => {
    const newId = "session_" + Date.now();
    const timeString = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const newSession = {
      id: newId,
      title: "New Chat",
      updatedAt: timeString,
      messages: [],
      datasetInfo: null,
    };

    setConversations((prev) => [newSession, ...prev]);
    setCurrentSessionId(newId);
    if (window.innerWidth <= 768) {
      setSidebarOpen(false);
    }
  };

  const selectChat = (id) => {
    setCurrentSessionId(id);
    if (window.innerWidth <= 768) {
      setSidebarOpen(false);
    }
  };

  const deleteChat = (id, e) => {
    e.stopPropagation();

    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      if (filtered.length === 0) {
        const newId = "session_" + Date.now();
        setCurrentSessionId(newId);
        return [
          {
            id: newId,
            title: "New Chat",
            updatedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            messages: [],
            datasetInfo: null,
          },
        ];
      }
      if (currentSessionId === id) {
        setCurrentSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", currentSessionId);

    setUploading(true);
    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to upload dataset.");

      const newDatasetInfo = data.dataset;

      // Update dataset metadata for current conversation
      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentSessionId
            ? { ...c, datasetInfo: newDatasetInfo }
            : c
        )
      );

      // Post notification message
      const colsList = newDatasetInfo.columns.join(", ");
      const notificationMsg = {
        role: "assistant",
        content: `📊 Dataset **"${newDatasetInfo.filename}"** uploaded successfully!\n\n• **Rows:** ${newDatasetInfo.rows}\n• **Columns:** ${colsList}\n\nYou can now ask me any questions about this dataset!`,
      };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentSessionId
            ? { ...c, messages: [...c.messages, notificationMsg] }
            : c
        )
      );
    } catch (err) {
      console.error("Upload error:", err);
      alert("Error uploading CSV dataset: " + err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const sendMessage = async (messageText = input) => {
    const message = messageText.trim();
    if (!message || loading) return;

    const userMsg = { role: "user", content: message };
    const updatedMessages = [...messages, userMsg];

    let sessionTitle = currentConversation?.title || "New Chat";
    if (sessionTitle === "New Chat" || messages.length === 0) {
      sessionTitle = message.length > 28 ? message.slice(0, 28) + "..." : message;
    }

    const timeString = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    setConversations((prev) =>
      prev.map((c) =>
        c.id === currentSessionId
          ? { ...c, title: sessionTitle, messages: updatedMessages, updatedAt: timeString }
          : c
      )
    );

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          session_id: currentSessionId,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Something went wrong.");

      const assistantMsg = { role: "assistant", content: data.response };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentSessionId
            ? {
                ...c,
                messages: [...c.messages, assistantMsg],
                updatedAt: timeString,
                datasetInfo: data.dataset || c.datasetInfo,
              }
            : c
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      const errorMsg = {
        role: "assistant",
        content: "Sorry, I couldn't process that request right now. Please try again later.",
      };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentSessionId
            ? { ...c, messages: [...c.messages, errorMsg], updatedAt: timeString }
            : c
        )
      );
    } finally {
      setLoading(false);
    }
  };

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
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".csv"
        onChange={handleFileUpload}
        style={{ display: "none" }}
      />

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="logo">ↂ</div>
            <span>AI Assistant</span>
          </div>

          <button
            className="new-chat-btn"
            onClick={createNewChat}
            title="Start new conversation"
          >
            <span>+</span> New Chat
          </button>
        </div>

        <div className="sidebar-history">
          <div className="history-label">Previous Conversations</div>
          <div className="history-list">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`history-item ${
                  conv.id === currentSessionId ? "active" : ""
                }`}
                onClick={() => selectChat(conv.id)}
              >
                <div className="history-icon">💬</div>
                <div className="history-details">
                  <div className="history-title">{conv.title}</div>
                  <div className="history-time">{conv.updatedAt}</div>
                </div>
                <button
                  className="delete-chat-btn"
                  onClick={(e) => deleteChat(conv.id, e)}
                  title="Delete chat"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="status-badge">● Online</div>
          <span>Upload CSV or ask questions</span>
        </div>
      </aside>

      {/* Main Chat Container */}
      <div className="chat-container">
        {/* Header */}
        <header className="chat-header">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            ☰
          </button>

          <div className="header-title">
            <div className="logo">ↂ</div>
            <div>
              <h1>{currentConversation?.title || "AI Assistant"}</h1>
              <p>Analyze data, calculate math, edit text, or search the web</p>
            </div>
          </div>

          {/* Dataset Status & Upload Button */}
          <div className="header-actions">
            {datasetInfo && (
              <div
                className="dataset-badge"
                title={`Columns: ${datasetInfo.columns.join(", ")}`}
              >
                📊 <span>{datasetInfo.filename}</span> ({datasetInfo.rows} rows)
              </div>
            )}

            <button
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              title="Upload your own CSV dataset"
            >
              📁 {uploading ? "Uploading..." : "Upload CSV"}
            </button>
          </div>
        </header>

        {/* Messages */}
        <main className="messages">
          {/* Welcome screen */}
          {messages.length === 0 && !loading && (
            <div className="welcome">
              <div className="welcome-icon">ↂ</div>
              <h2>What can I help you with?</h2>
              <p>
                Upload your CSV dataset or ask a question to analyze data, calculate math, edit text, or search the web.
              </p>

              {datasetInfo && (
                <div className="active-dataset-card">
                  <strong>Active Dataset:</strong> {datasetInfo.filename} ({datasetInfo.rows} rows)
                  <br />
                  <small>Columns: {datasetInfo.columns.join(", ")}</small>
                </div>
              )}

              <div className="examples">
                <button
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  <strong>Upload CSV Data</strong>
                  <span>Analyze your own spreadsheet/CSV file</span>
                </button>

                <button
                  onClick={() =>
                    handleExampleClick(
                      datasetInfo
                        ? `Summarize the columns in ${datasetInfo.filename}`
                        : "Which city has the highest sales?"
                    )
                  }
                >
                  <strong>Analyze Data</strong>
                  <span>
                    {datasetInfo
                      ? `Analyze ${datasetInfo.filename}`
                      : "Which city has the highest sales?"}
                  </span>
                </button>

                <button
                  onClick={() => handleExampleClick("What is 15% of 850?")}
                >
                  <strong>Advanced Math</strong>
                  <span>What is 15% of 850?</span>
                </button>

                <button
                  onClick={() =>
                    handleExampleClick(
                      "Rewrite professionally: hey bro please send me the report ASAP"
                    )
                  }
                >
                  <strong>Text Utilities</strong>
                  <span>Rewrite professionally: send report ASAP</span>
                </button>
              </div>
            </div>
          )}

          {/* Chat messages */}
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === "assistant" && (
                <div className="message-avatar">ↂ</div>
              )}
              <div className="message-bubble">{message.content}</div>
            </div>
          ))}

          {/* Loading animation */}
          {loading && (
            <div className="message assistant">
              <div className="message-avatar">ↂ</div>
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
            <button
              className="input-upload-icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              title="Upload CSV dataset"
            >
              📎
            </button>
            <input
              type="text"
              placeholder={
                datasetInfo
                  ? `Ask a question about ${datasetInfo.filename}...`
                  : "Ask me anything or upload a CSV dataset..."
              }
              value={input}
              onChange={(event) => setInput(event.target.value)}
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