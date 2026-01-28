import { useEffect, useState } from "react";
import Card from "react-bootstrap/Card";
import ListGroup from "react-bootstrap/ListGroup";
import Placeholder from "react-bootstrap/Placeholder";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function News() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let alive = true;

    async function load() {
      setLoading(true);
      setErr("");
      try {
        const res = await fetch(`${API_BASE}/news/latest?limit=20`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!alive) return;
        setItems(data.items || []);
      } catch (e) {
        if (!alive) return;
        setErr(e?.message || "Failed to load news");
        setItems([]);
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  const openNews = (link) => {
    if (!link) return;
    window.open(link, "_blank", "noopener,noreferrer");
  };

  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: "24px 16px" }}>
      <h2 style={{ marginBottom: 16 }}>뉴스</h2>

      {err ? (
        <div style={{ marginBottom: 12, color: "crimson" }}>
          불러오기 실패: {err}
        </div>
      ) : null}

      {loading ? (
        <Card>
          <Card.Body>
            <Placeholder as={Card.Title} animation="glow">
              <Placeholder xs={8} />
            </Placeholder>
            <Placeholder as={Card.Text} animation="glow">
              <Placeholder xs={12} /> <Placeholder xs={11} /> <Placeholder xs={10} />
            </Placeholder>
            <Placeholder as={Card.Text} animation="glow">
              <Placeholder xs={12} />
            </Placeholder>
          </Card.Body>
        </Card>
      ) : (
        <ListGroup>
          {items.map((it, idx) => (
            <ListGroup.Item key={`${it.link}-${idx}`} style={{ padding: 0 }}>
              <Card style={{ border: "none" }}>
                <Card.Body>
                  <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                    <Card.Title style={{ margin: 0, fontSize: 18 }}>
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={() => openNews(it.link)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") openNews(it.link);
                        }}
                        style={{
                          cursor: "pointer",
                          textDecoration: "underline",
                        }}
                        title="새 탭에서 열기"
                      >
                        {it.title}
                      </span>
                    </Card.Title>

                    <div style={{ marginLeft: "auto", fontSize: 12, opacity: 0.75 }}>
                      {it.press ? `${it.press} · ` : ""}
                      {it.date || it.date_iso || ""}
                    </div>
                  </div>

                  <div style={{ marginTop: 10, whiteSpace: "pre-line", lineHeight: 1.5 }}>
                    {(it.preview_lines || []).length ? (
                      it.preview_lines.map((ln, i) => (
                        <div key={i}>{ln}</div>
                      ))
                    ) : (
                      <div style={{ opacity: 0.7 }}>미리보기 없음</div>
                    )}
                  </div>

                  <div
                    style={{
                      marginTop: 10,
                      paddingTop: 10,
                      borderTop: "1px solid rgba(0,0,0,0.08)",
                      opacity: 0.9,
                    }}
                  >
                    <strong>요약:</strong> {it.summary || "요약 없음"}
                  </div>
                </Card.Body>
              </Card>
            </ListGroup.Item>
          ))}
        </ListGroup>
      )}
    </div>
  );
}

export default News;