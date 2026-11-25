
import { useState } from "react";
import "./App.css";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!review.trim()) {
      alert("Please enter a review");
      return;
    }

    setLoading(true);
    setResult("");

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze review");
      }

      const data = await response.json();
      setResult(data.predicted_sentiment);
    } catch (error) {
      setResult("Error connecting to API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Amazon Sentiment Analysis</h1>

        <label>Enter the Review</label>

        <textarea
          placeholder="Write your product review here..."
          value={review}
          onChange={(e) => setReview(e.target.value)}
        />

        <button onClick={analyzeSentiment} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>

        {result !== "" && (
          <div className="result">
            <h2>Result</h2>
            <p>{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
