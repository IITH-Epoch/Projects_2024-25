"use client";
import { useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      const response = await axios.post("/api/register", { email, password });
      setSuccess("Registration successful! Redirecting to login...");
      setError("");
      setTimeout(() => {
        router.push("/login");
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.message || "Registration failed.");
      setSuccess("");
    }
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.blurLayer}></div>
      <div style={styles.overlay}></div>
      <div style={styles.header}>IITH GPT</div>
      <div style={styles.card}>
        <h2 style={styles.title}>Register</h2>
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Email:</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Password:</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Confirm Password:</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          {error && <p style={styles.error}>{error}</p>}
          {success && <p style={styles.success}>{success}</p>}
          <button type="submit" style={styles.button}>
            Register
          </button>
        </form>
        <p style={styles.linkText}>
          Already have an account?{" "}
          <Link href="/login" style={styles.link}>
            Login
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
    backgroundColor:"black"
  },

  // blurLayer: {
  //   position: "absolute",
  //   top: 0,
  //   left: 0,
  //   width: "100%",
  //   height: "100%",
  //   backgroundImage: `url("data:image/jpeg;base64,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")`,
  //   backgroundSize: "cover",
  //   backgroundPosition: "center",
  //   backgroundRepeat: "no-repeat",
  //   filter: "blur(5px)",
  //   zIndex: -1, // Keeps the blur behind the login card
  // },
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
    background: "rgba(255, 255, 255, 0)",
    backdropFilter: "blur(10px)",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.3)",
    textAlign: "center",
    position: "relative",
    zIndex: 1,
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
  linkText: {
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