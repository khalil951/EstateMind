import { NavLink, Route, Routes } from "react-router-dom";
import ListingsPage from "./pages/ListingsPage.tsx";
import ValuationPage from "./pages/ValuationPage.tsx";

export default function App() {
  return (
    <div className="app-shell">
      <div className="ambient-layer" aria-hidden="true">
        <span className="orb orb-one" />
        <span className="orb orb-two" />
        <span className="grid-overlay" />
      </div>
      <header className="topbar">
        <div className="brand-block">
          <p className="section-label">INTELLIGENCE WORKSPACE</p>
          <h1>EstateMind</h1>
        </div>
        <nav className="topnav">
          <NavLink to="/listings" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Explore Listings
          </NavLink>
          <NavLink to="/valuation" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Valuation Lab
          </NavLink>
        </nav>
      </header>
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<ListingsPage />} />
          <Route path="/listings" element={<ListingsPage />} />
          <Route path="/valuation" element={<ValuationPage />} />
        </Routes>
      </main>
    </div>
  );
}
