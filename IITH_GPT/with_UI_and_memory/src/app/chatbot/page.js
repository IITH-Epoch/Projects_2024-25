"use client";
import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRouter } from "next/navigation";
import { FaMicrophone, FaMicrophoneSlash, FaUpload, FaFileAlt } from "react-icons/fa";

export default function Chatbot() {
  const router = useRouter();
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [userEmail, setUserEmail] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);

  // State for image upload (OCR)
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [ocrLoading, setOcrLoading] = useState(false);

  // State for document upload
  const [selectedDocument, setSelectedDocument] = useState(null);

  // Retrieve user email from localStorage
  useEffect(() => {
    const email = localStorage.getItem("email");
    if (!email) {
      router.push("/login");
    } else {
      setUserEmail(email);
    }
  }, [router]);

  // Function to speak text using the Web Speech API
  const speakText = (text) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
    } else {
      console.warn("Speech synthesis not supported in this browser.");
    }
  };

  // Single send function: if an image or document is selected, process accordingly;
  // otherwise, send the text message.
  const handleSend = async () => {
    if (selectedImage) {
      // Append the user message with image preview and optional question text
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: userInput, sender: "user", image: imagePreview },
      ]);
      await handleProcessImage();
      setUserInput("");
      return;
    }
    if (selectedDocument) {
      // Append a message noting that a document was uploaded (display file name)
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: userInput || `Uploaded Document: ${selectedDocument.name}`, sender: "user" },
      ]);
      await handleProcessDocument();
      setUserInput("");
      return;
    }
    if (userInput.trim() === "") return;
    const newMessages = [...messages, { text: userInput, sender: "user" }];
    setMessages(newMessages);
    setUserInput("");

    setLoading(true);
    try {
      const botResponse = await getBotResponse(userInput, newMessages);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: botResponse, sender: "bot" },
      ]);
      if (ttsEnabled) {
        speakText(botResponse);
      }
    } catch (error) {
      console.error("Error fetching bot response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: "Oops! Something went wrong. Please try again later.",
          sender: "bot",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Call backend API for bot response
  const getBotResponse = async (input, memory) => {
    const emailToSend = userEmail || localStorage.getItem("email") || "";
    const response = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: input.trim(), email: emailToSend, memory }),
    });
    if (!response.ok) {
      throw new Error("Failed to get response from server");
    }
    const data = await response.json();
    return data.response;
  };

  // Handle speech input
  const handleSpeechInput = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Sorry, your browser does not support speech recognition.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsRecording(true);
    };

    recognition.onresult = (event) => {
      const speechToText = event.results[0][0].transcript;
      processSpeechToText(speechToText);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognition.start();
  };

  // Process recognized speech as user query
  const processSpeechToText = async (speechText) => {
    const newMessages = [...messages, { text: speechText, sender: "user" }];
    setMessages(newMessages);

    setLoading(true);
    try {
      const botResponse = await getBotResponse(speechText, newMessages);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: botResponse, sender: "bot" },
      ]);
      if (ttsEnabled) {
        speakText(botResponse);
      }
    } catch (error) {
      console.error("Error fetching bot response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: "Oops! Something went wrong. Please try again later.",
          sender: "bot",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto-scroll chat box to bottom on message update
    const chatBox = document.querySelector("[style*='overflowY: scroll']");
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);

  // Handle file selection for OCR and create a preview URL for images
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  // Process the selected image: send it to your backend for OCR and markdown conversion,
  // including the question (if provided)
  const handleProcessImage = async () => {
    if (!selectedImage) return;
    setOcrLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", selectedImage);
      if (userInput.trim() !== "") {
        formData.append("question", userInput.trim());
      }
      const response = await fetch("http://localhost:5000/upload-image", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to process image");
      }
      const data = await response.json();
      // Append OCR output (markdown) as a new bot message
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: data.markdown, sender: "bot" },
      ]);
      setSelectedImage(null);
      setImagePreview(null);
    } catch (error) {
      console.error("Error processing image:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Error processing image.", sender: "bot" },
      ]);
    } finally {
      setOcrLoading(false);
    }
  };

  // Process the selected document: send it to your backend for summarization
  const handleProcessDocument = async () => {
    if (!selectedDocument) return;
    setOcrLoading(true);
    try {
      const formData = new FormData();
      formData.append("document", selectedDocument);
      const emailToSend = userEmail || localStorage.getItem("email") || "";
      formData.append("email", emailToSend);
      if (userInput.trim() !== "") {
        formData.append("question", userInput.trim());
      }
      const response = await fetch("http://localhost:5000/upload-document", {
        method: "POST",
        body: formData,
      });
      console.log("Document upload response status:", response.status);
      const data = await response.json();
      console.log("Response data from upload-document:", data);
      
      // Append the returned markdown (if available) as a bot message
      if (data.markdown) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: data.markdown, sender: "bot" },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: "No markdown output received.", sender: "bot" },
        ]);
      }
      setSelectedDocument(null);
    } catch (error) {
      console.error("Error processing document:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Error processing document.", sender: "bot" },
      ]);
    } finally {
      setOcrLoading(false);
    }
  };

  // Handle file selection for document upload
  const handleDocumentChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedDocument(e.target.files[0]);
    }
  };
  

  // Custom markdown rendering components to preserve your headings, list styles, etc.
  const markdownComponents = {
    h1: ({ children }) => (
      <h1 style={{ fontSize: "2rem", marginBottom: "15px", borderBottom: "1px solid #444", paddingBottom: "5px", color: "#fff" }}>
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 style={{ fontSize: "1.8rem", marginBottom: "10px", color: "#fff" }}>
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 style={{ fontSize: "1.6rem", marginBottom: "8px", color: "#fff" }}>
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p style={{ margin: "10px 0", lineHeight: "1.6", color: "#eee" }}>
        {children}
      </p>
    ),
    ul: ({ children }) => (
      <ul style={{ margin: "10px 0", paddingLeft: "20px", color: "#ccc" }}>
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol style={{ margin: "10px 0", paddingLeft: "20px", color: "#ccc" }}>
        {children}
      </ol>
    ),
    li: ({ children }) => (
      <li style={{ margin: "5px 0", color: "#ddd" }}>
        {children}
      </li>
    ),
    strong: ({ children }) => (
      <strong style={{ fontWeight: "bold", color: "#fff" }}>
        {children}
      </strong>
    ),
    em: ({ children }) => (
      <em style={{ fontStyle: "italic", color: "#fff" }}>
        {children}
      </em>
    ),
    del: ({ children }) => (
      <del style={{ textDecoration: "line-through", color: "#bbb" }}>
        {children}
      </del>
    ),
    blockquote: ({ children }) => (
      <blockquote style={{ borderLeft: "4px solid #888", paddingLeft: "10px", color: "#aaa", fontStyle: "italic" }}>
        {children}
      </blockquote>
    ),
    code: ({ children }) => (
      <code style={{ backgroundColor: "#222", padding: "2px 4px", borderRadius: "4px", color: "#0f0" }}>
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre style={{ backgroundColor: "#111", padding: "10px", borderRadius: "5px", overflowX: "auto", color: "#0f0" }}>
        {children}
      </pre>
    ),
    hr: () => <hr style={{ border: "none", borderTop: "1px solid #555", margin: "15px 0" }} />,
    a: ({ children, href }) => (
      <a href={href} style={{ color: "#4af", textDecoration: "underline" }} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ),
    img: ({ src, alt }) => (
      <img src={src} alt={alt} style={{ maxWidth: "100%", borderRadius: "5px" }} />
    ),
    table: ({ children }) => (
      <table style={{ borderCollapse: "collapse", width: "100%", margin: "15px 0", color: "#eee" }}>
        {children}
      </table>
    ),
    th: ({ children }) => (
      <th style={{ border: "1px solid #555", padding: "8px", backgroundColor: "#222", color: "#fff" }}>
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td style={{ border: "1px solid #555", padding: "8px", color: "#ccc" }}>
        {children}
      </td>
    )
  };

  return (
    <div style={styles.container}>
      <div style={styles.header1}>IITH GPT</div>
      <div style={styles.header}>
        <h3 style={styles.headerText}>Welcome back</h3>
      </div>
      
      <div style={styles.ttsContainer}>
        <input
          type="checkbox"
          id="ttsToggle"
          checked={ttsEnabled}
          onChange={(e) => setTtsEnabled(e.target.checked)}
        />
        <label htmlFor="ttsToggle" style={styles.ttsLabel}>
          Read responses aloud
        </label>
      </div>
      
      <div style={styles.chatBox}>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.message,
              alignSelf: message.sender === "user" ? "flex-end" : "flex-start",
              backgroundColor: message.sender === "user" ? "#444" : "#222",
              borderRadius: message.sender === "user" ? "15px 15px 0px 15px" : "15px 15px 15px 0px",
            }}
          >
            <strong>{message.sender === "user" ? "You: " : "IITH_GPT: "}</strong>
            {message.image ? (
              <>
                <img src={message.image} alt="uploaded" style={styles.uploadedImage} />
                {message.text && (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {message.text}
                  </ReactMarkdown>
                )}
              </>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {message.text}
              </ReactMarkdown>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ ...styles.message, alignSelf: "flex-start" }}>
            Typing...
          </div>
        )}
      </div>
      
      <div style={styles.inputContainer}>
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type your message..."
          style={styles.input}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={loading}
        />
        {/* Show image preview if available */}
        {imagePreview && (
          <img src={imagePreview} alt="Preview" style={styles.previewImage} />
        )}
        {/* Upload Image button */}
        <label style={styles.uploadButton}>
          <FaUpload />
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: "none" }}
          />
        </label>
        {/* Upload Document button */}
        <label style={styles.uploadButton}>
          <FaFileAlt />
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={handleDocumentChange}
            style={{ display: "none" }}
          />
        </label>
        {selectedDocument && (
          <span style={{ color: "#fff", marginLeft: "5px" }}>{selectedDocument.name}</span>
        )}
        <button
          onClick={handleSend}
          style={styles.button}
          disabled={loading || (userInput.trim() === "" && !selectedImage && !selectedDocument)}
        >
          {loading || ocrLoading ? "Sending..." : "Send"}
        </button>
        <button
          onClick={handleSpeechInput}
          style={styles.button}
          disabled={isRecording || loading}
        >
          {isRecording ? (
            <FaMicrophoneSlash style={{ fontSize: "20px", color: "#ff4d4d" }} />
          ) : (
            <FaMicrophone style={{ fontSize: "20px", color: "#4dff4d" }} />
          )}
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    width: "100vw",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#000",
    padding: "0",
    margin: "0",
  },
  header: {
    width: "100%",
    padding: "15px 20px",
    textAlign: "center",
    color: "#fff",
    fontSize: "1.5rem",
    fontWeight: "bold",
    backgroundColor: "#000",
  },
  header1: {
    position: "absolute",
    top: "20px",
    left: "50px",
    fontSize: "1.2rem",
    fontWeight: "bold",
    color: "white",
    zIndex: 2,
  },
  ttsContainer: {
    display: "flex",
    alignItems: "center",
    margin: "10px 0",
    color: "#fff",
    fontSize: "16px",
  },
  ttsLabel: {
    marginLeft: "5px",
  },
  chatBox: {
    width: "70vw",
    height: "75vh",
    padding: "20px",
    overflowY: "scroll",
    display: "flex",
    flexDirection: "column",
    gap: "15px",
    backgroundColor: "transparent",
    scrollbarWidth: "thin",
    scrollbarColor: "#444 #222",
  },
  message: {
    maxWidth: "70%",
    padding: "12px",
    borderRadius: "12px",
    fontSize: "16px",
    wordWrap: "break-word",
    color: "#fff",
  },
  inputContainer: {
    display: "flex",
    width: "70vw",
    marginTop: "10px",
    alignItems: "center",
    gap: "5px",
  },
  input: {
    flex: 1,
    padding: "15px",
    border: "none",
    borderRadius: "8px 0 0 8px",
    outline: "none",
    fontSize: "16px",
    backgroundColor: "#444",
    color: "#fff",
  },
  button: {
    padding: "15px 20px",
    background: "linear-gradient(to right, #444, #222)",
    color: "white",
    border: "none",
    borderRadius: "0 8px 8px 0",
    cursor: "pointer",
    fontSize: "16px",
    transition: "background 0.3s ease",
    marginLeft: "5px",
  },
  uploadButton: {
    padding: "15px 20px",
    background: "linear-gradient(to right, #444, #222)",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
    transition: "background 0.3s ease",
    marginLeft: "5px",
    display: "inline-block",
  },
  previewImage: {
    height: "50px",
    width: "auto",
    borderRadius: "4px",
    marginLeft: "5px",
    border: "1px solid #fff",
  },
  uploadedImage: {
    maxHeight: "100px",
    marginBottom: "5px",
    borderRadius: "4px",
  },
};

const markdownStyles = {
  fontFamily: "Arial, sans-serif",
  fontSize: "16px",
  color: "#fff",
  lineHeight: "1.5",
  marginTop: "5px",
};
