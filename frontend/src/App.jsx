import { Routes, Route, useNavigate, Outlet, NavLink } from "react-router-dom";
import { LineChart } from "lucide-react";

import Chatbot from "./Chatbot";
import News from "./News";

import "./App.css";

function App() {
  return (
    <>
      <Routes>
        {/* 네비게이션바가 있는 레이아웃 안에 들어가는 페이지들 */}
        <Route element={<LayoutWithNavbar />}>
          <Route path="/" element={<Chatbot />} />
          <Route path="/chat" element={<Chatbot />} />
          <Route path="/news" element={<News />} />
        </Route>

        {/* 네비게이션바 없는 404 */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}

export default App;

function LayoutWithNavbar() {
  const navigate = useNavigate();

  return (
    <>
      <nav className="navbar">
        <div className="navbar-container">
          <div
            className="navbar-brand"
            onClick={() => {
              navigate("/");
            }}
          >
            {/* <span className="brand-icon">📈</span> */}
            <span className="brand-icon-svg" aria-hidden="true">
              <LineChart size={20} strokeWidth={2.4} />
            </span>
            <span className="brand-text">Stock Mate</span>
          </div>
          <div className="navbar-menu">
            <button
              className="nav-link"
              onClick={() => {
                navigate("/chat");
              }}
            >
              Chat
            </button>
            <button
              className="nav-link"
              onClick={() => {
                navigate("/news");
              }}
            >
              News
            </button>
          </div>
        </div>
      </nav>
      {/* </CHANGE> */}

      {/* 여기서 각 페이지가 바뀌어 들어감 */}
      <Outlet />
    </>
  );
}

// 404 전용 컴포넌트 (네비게이션바 없음)
function NotFound() {
  return (
    <div className="not-found">
      <h1>404</h1>
      <p>페이지를 찾을 수 없습니다</p>
    </div>
  );
}
