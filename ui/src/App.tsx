import { useState } from "react";

const API = "http://localhost:8000";

function ScoreBar({ score }) {
  const max = 10;
  const pct = Math.min((score / max) * 100, 100);
  const color = score > 6 ? "#00ff87" : score > 3 ? "#ffbe0b" : "#ff006e";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 4, background: "#1a1a1a",
        borderRadius: 2, overflow: "hidden"
      }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ color, fontFamily: "monospace", fontSize: 12, minWidth: 36 }}>
        {score.toFixed(2)}
      </span>
    </div>
  );
}

function ChunkCard({ chunk, index }) {
  const [expanded, setExpanded] = useState(false);
  const isRelevant = chunk.score > 0;

  return (
    <div style={{
      background: "#0d0d0d", border: "1px solid #222",
      borderRadius: 8, padding: 16, cursor: "pointer",
      transition: "border-color 0.2s"
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = "#444"}
      onMouseLeave={e => e.currentTarget.style.borderColor = "#222"}
      onClick={() => setExpanded(!expanded)}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{
            background: "#1a1a2e", color: "#818cf8",
            padding: "2px 8px", borderRadius: 4,
            fontFamily: "monospace", fontSize: 11, fontWeight: 700
          }}>
            S{String(chunk.season).padStart(2, "0")}E{chunk.episode}
          </span>
          <span style={{ color: "#555", fontSize: 11 }}>chunk {index + 1}</span>
        </div>
        <span style={{ color: "#555", fontSize: 11 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      <ScoreBar score={chunk.score} />
      <span style={{
  color: chunk.score > 5 ? "#00ff87" : chunk.score > 0 ? "#ffbe0b" : "#ff006e",
  fontSize: 10,
  fontFamily: "monospace",
  marginTop: 4,
  display: "block"
}}>
  {chunk.score > 5 ? "HIGH RELEVANCE" : chunk.score > 0 ? "MEDIUM" : "LOW — likely wrong chunk"}
</span>
      {expanded && (
        <pre style={{
          marginTop: 12, color: "#aaa", fontSize: 12,
          whiteSpace: "pre-wrap", fontFamily: "monospace",
          lineHeight: 1.6, borderTop: "1px solid #1a1a1a", paddingTop: 12
        }}>
          {chunk.text}
        </pre>
      )}
    </div>
  );
}

function MetricsBar({ metrics }) {
  const steps = [
    { label: "Vector", ms: metrics.vector_search_ms, color: "#818cf8" },
    { label: "BM25", ms: metrics.bm25_search_ms, color: "#34d399" },
    { label: "Rerank", ms: metrics.rerank_ms, color: "#fbbf24" },
    { label: "LLM", ms: metrics.llm_ms, color: "#f87171" },
  ];
  return (
    <div style={{ background: "#0d0d0d", border: "1px solid #222", borderRadius: 8, padding: 16 }}>
      <div style={{ color: "#555", fontSize: 11, marginBottom: 12, fontFamily: "monospace" }}>
        LATENCY BREAKDOWN — total {metrics.total_ms}ms
      </div>
      <div style={{ display: "flex", gap: 4, height: 24, borderRadius: 4, overflow: "hidden" }}>
        {steps.map(s => {
          const pct = (s.ms / metrics.total_ms) * 100;
          return (
            <div key={s.label} style={{
              width: `${pct}%`, background: s.color,
              display: "flex", alignItems: "center",
              justifyContent: "center", fontSize: 10,
              color: "#000", fontWeight: 700, fontFamily: "monospace",
              minWidth: s.ms > 50 ? "auto" : 0
            }}>
              {s.ms > 100 ? `${s.ms}ms` : ""}
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
        {steps.map(s => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: s.color }} />
            <span style={{ color: "#555", fontSize: 11, fontFamily: "monospace" }}>
              {s.label} {s.ms}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentStep({ step, index }) {
  const [expanded, setExpanded] = useState(false);
  const isCall = step.type === "tool_call";
  return (
    <div style={{
      borderLeft: `2px solid ${isCall ? "#818cf8" : "#34d399"}`,
      paddingLeft: 12, marginBottom: 8
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        cursor: "pointer", padding: "4px 0"
      }} onClick={() => setExpanded(!expanded)}>
        <span style={{
          color: isCall ? "#818cf8" : "#34d399",
          fontFamily: "monospace", fontSize: 11
        }}>
          {isCall ? "→ CALL" : "← RESULT"}
        </span>
        <span style={{ color: "#aaa", fontFamily: "monospace", fontSize: 12, fontWeight: 700 }}>
          {step.tool}
        </span>
        <span style={{ color: "#555", fontSize: 11 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <pre style={{
          color: "#666", fontSize: 11, fontFamily: "monospace",
          whiteSpace: "pre-wrap", lineHeight: 1.5,
          background: "#0a0a0a", padding: 8, borderRadius: 4, marginTop: 4
        }}>
          {JSON.stringify(isCall ? step.input : step.output, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("rag");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!question.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const endpoint = mode === "rag" ? "/ask" : "/agent/ask";
      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError("Failed to connect to backend. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      minHeight: "100vh", background: "#080808", color: "#e0e0e0",
      fontFamily: "'Courier New', monospace", padding: "40px 24px",
      maxWidth: 900, margin: "0 auto"
    }}>
      {/* header */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ color: "#555", fontSize: 11, letterSpacing: 4, marginBottom: 8 }}>
          FRIENDS SCRIPT INTELLIGENCE
        </div>
        <h1 style={{
          fontSize: 32, fontWeight: 900, margin: 0,
          color: "#fff", letterSpacing: -1
        }}>
          RAG Debug UI
        </h1>
        <div style={{ color: "#444", fontSize: 12, marginTop: 4 }}>
          inspect retrieval · trace agents · measure quality
        </div>
      </div>

      {/* mode toggle */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        {["rag", "agent"].map(m => (
          <button key={m} onClick={() => setMode(m)} style={{
            padding: "6px 16px", borderRadius: 4, cursor: "pointer",
            fontFamily: "monospace", fontSize: 12, fontWeight: 700,
            border: "1px solid",
            borderColor: mode === m ? "#818cf8" : "#333",
            background: mode === m ? "#1a1a2e" : "transparent",
            color: mode === m ? "#818cf8" : "#555",
            transition: "all 0.2s"
          }}>
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      {/* input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 32 }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSubmit()}
          placeholder={mode === "rag"
            ? "What is the Moist Maker?"
            : "Compare how Joey and Ross talk about their careers..."
          }
          style={{
            flex: 1, background: "#0d0d0d", border: "1px solid #333",
            borderRadius: 6, padding: "12px 16px", color: "#e0e0e0",
            fontFamily: "monospace", fontSize: 14, outline: "none"
          }}
        />
        <button onClick={handleSubmit} disabled={loading} style={{
          padding: "12px 24px", background: loading ? "#1a1a1a" : "#818cf8",
          color: loading ? "#555" : "#000", border: "none", borderRadius: 6,
          cursor: loading ? "not-allowed" : "pointer",
          fontFamily: "monospace", fontWeight: 700, fontSize: 13,
          transition: "all 0.2s"
        }}>
          {loading ? "THINKING..." : "ASK →"}
        </button>
      </div>

      {error && (
        <div style={{ color: "#ff006e", fontFamily: "monospace", fontSize: 13, marginBottom: 24 }}>
          ✗ {error}
        </div>
      )}

      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

          {/* answer */}
          <div style={{ background: "#0d0d0d", border: "1px solid #222", borderRadius: 8, padding: 20 }}>
            <div style={{ color: "#555", fontSize: 11, marginBottom: 12, letterSpacing: 2 }}>
              ANSWER
            </div>
            <p style={{ color: "#e0e0e0", lineHeight: 1.8, margin: 0, fontSize: 15 }}>
              {result.answer}
            </p>
          </div>

          {/* RAG mode — chunks + metrics */}
          {mode === "rag" && result.retrieved_chunks && (
            <>
              <div>
                <div style={{ color: "#555", fontSize: 11, marginBottom: 12, letterSpacing: 2 }}>
                  RETRIEVED CHUNKS ({result.retrieved_chunks.length})
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {result.retrieved_chunks.map((chunk, i) => (
                    <ChunkCard key={i} chunk={chunk} index={i} />
                  ))}
                </div>
              </div>
              {result.metrics && <MetricsBar metrics={result.metrics} />}
            </>
          )}

          {/* Agent mode — steps */}
          {mode === "agent" && result.steps && (
            <div>
              <div style={{ color: "#555", fontSize: 11, marginBottom: 12, letterSpacing: 2 }}>
                AGENT TRACE — {result.total_steps} steps
              </div>
              {result.steps.map((step, i) => (
                <AgentStep key={i} step={step} index={i} />
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  );
}