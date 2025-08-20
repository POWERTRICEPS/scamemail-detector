import { useState } from "react";

function App() {

  const [email, setEmail] = useState("");
  const [result, setResult] = useState(null);
  

  const handleSubmit = async () => {
    const response = await fetch("http://localhost:5000/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: email }),
    });

    const data = await response.json();
    setResult(data);
  };


  
  return (
    <div
    style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh"
    }}
  >
    <div style={{ textAlign: "center" }}>
      <h1>Scam Email Detector</h1>
      <textarea
        rows="20"
        cols="100"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Paste your email text here..."
      />
      <br />
      <button onClick={handleSubmit}>Check Email</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2>Result: {result.label}</h2>
          <p>
            Scam Probability: {(result.scam_probability * 100).toFixed(0)}%
          </p>
        </div>
      )}
    </div>
  </div>
);
}


export default App;
