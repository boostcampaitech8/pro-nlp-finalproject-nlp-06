import { useEffect, useState } from "react";
import Card from "react-bootstrap/Card";
import Placeholder from "react-bootstrap/Placeholder";
import "./news.css";

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
    <div className="news-page">
      <h2 className="news-page-title">뉴스</h2>

      {err ? <div className="news-error">불러오기 실패: {err}</div> : null}

      {loading ? (
        <div className="news-grid">
          {[0, 1, 2].map((n) => (
            <Card key={n} className="news-card">
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
          ))}
        </div>
      ) : (
        <div className="news-grid">
          {items.map((it, idx) => (
            <Card key={`${it.link || "no-link"}-${idx}`} className="news-card">
              <Card.Body className="news-card-body">
                <div className="news-card-header">
                    <Card.Title className="news-title">{it.title || "제목 없음"}</Card.Title>
                    <button
                        type="button"
                        className="news-open-btn"
                        onClick={() => openNews(it.link)}
                        disabled={!it.link}
                        aria-label={`원문 보기: ${it.title || "뉴스"}`}
                    >
                        원문 보기
                    </button>
                    </div>
                    <div className="news-meta-row">
                    <div className="news-meta">
                        {it.press ? `${it.press} · ` : ""}
                        {it.date || it.date_iso || ""}
                    </div>
                </div>

                <div className="news-preview">
                  {(it.preview_lines || []).length ? (
                    it.preview_lines.map((ln, i) => (
                      <p key={i} className="news-preview-line">
                        {ln}
                      </p>
                    ))
                  ) : (
                    <p className="news-preview-empty">미리보기 없음</p>
                  )}
                </div>

                <div className="news-summary">
                  <strong className="news-summary-label">요약</strong>
                  <p className="news-summary-text">{it.summary || "요약 없음"}</p>
                </div>
              </Card.Body>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

export default News;
