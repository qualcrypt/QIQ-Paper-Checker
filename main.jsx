import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import PaperChecker from "./PaperChecker.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <PaperChecker />
  </StrictMode>
);
