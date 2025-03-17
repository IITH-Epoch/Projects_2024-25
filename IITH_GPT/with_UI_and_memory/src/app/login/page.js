"use client";
import { useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("/api/login", { email, password });
      const token = response.data.token;
      localStorage.setItem("token", token);
      localStorage.setItem("email", email);
      router.push("/");
    } catch (err) {
      setError(err.response?.data?.message || "Login failed.");
    }
  };

  return (
    <div style={styles.wrapper}>
      {/* Blurred Background Layer */}
      <div style={styles.header}>IITH GPT</div>
      <div style={styles.blurLayer}></div>
      <div style={styles.overlay}></div>
      <div style={styles.card}>
        <h2 style={styles.title}>Sign in</h2>
        <p style={styles.subtitle}>Welcome Back</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          {error && <p style={styles.error}>{error}</p>}
          <button type="submit" style={styles.button}>
            Sign In
          </button>
        </form>

        <p style={styles.registerText}>
          <span style={{ color: "white" }}>Don't have an account?</span>{" "}
          <Link href="/register" style={styles.link}>
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    minHeight: "100vh",
    margin: 0,
    padding: 0,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "'Inter', sans-serif",
    position: "relative",
    overflow: "hidden",
  },

  blurLayer: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundImage: `url("data:image/jpeg;base64,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")`,
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    filter: "blur(5px)",
    zIndex: -1, // Keeps the blur behind the login card
  },

  header: {
    position: "absolute",
    top: "20px",
    left: "50px",
    fontSize: "2rem",
    fontWeight: "bold",
    color: "white",
    zIndex: 2,  // Ensure it's above other elements
  },

  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.6)",
  },
  card: {
    width: "400px",
    padding: "2rem",
    borderRadius: "16px",
    background: "black",
    backdropFilter: "blur(10px)",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.3)",
    textAlign: "center",
    position: "relative",
    zIndex: 1,
  },

  subtitle: {
    marginBottom: "2rem",
    fontSize: "1.2rem",
    color: "rgb(234, 0, 51)",
    fontWeight: "800",
  },

  title: {
    marginBottom: "1rem",
    fontSize: "1.8rem",
    color: "#fff",
    fontWeight: "750",
  },
  form: {
    display: "flex",
    flexDirection: "column",
  },
  formGroup: {
    marginBottom: "1rem",
    textAlign: "left",
  },
  label: {
    display: "block",
    marginBottom: "0.5rem",
    color: "#ddd",
    fontSize: "0.9rem",
  },
  input: {
    width: "100%",
    padding: "0.75rem",
    borderRadius: "8px",
    border: "none",
    outline: "none",
    fontSize: "0.95rem",
    background: "rgba(255, 255, 255, 0.2)",
    color: "white",
    fontWeight: "600",
    boxShadow: "inset 0 2px 5px rgba(255, 255, 255, 0.2)",
  },
  button: {
    marginTop: "1rem",
    padding: "0.75rem",
    border: "none",
    borderRadius: "8px",
    fontSize: "1rem",
    fontWeight: "600",
    color: "white",
    cursor: "pointer",
    background: "linear-gradient(to right, #ff416c, #ff4b2b)",
    boxShadow: "0 4px 15px rgba(255, 64, 75, 0.4)",
    transition: "background 0.3s ease",
  },

  error: {
    color: "#ff4f4f",
    fontSize: "0.9rem",
    margin: "0.5rem 0",
  },
  registerText: {
    textAlign: "center",
    marginTop: "1rem",
    fontSize: "0.9rem",
    color: "#ccc",
  },

  link: {
    color: "#ff416c",
    textDecoration: "none",
    fontWeight: "600",
    marginLeft: "0.25rem",
  },
};
