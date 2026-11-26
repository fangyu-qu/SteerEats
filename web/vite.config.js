// Relative base: the built site works from any folder, e.g. https://fangyu-qu.github.io/SteerEats/ on GitHub
// Pages, a Netlify root, or a local static server. Data files are fetched relative to the page (src/main.js).
import { defineConfig } from "vite";

// In development /api goes to the local API (server/app.py on port 8787); the built site has no proxy.
// Two pages: index.html is the diner app, lab.html the results dashboard and explorer.
export default defineConfig({
  base: "./",
  server: { proxy: { "/api": "http://127.0.0.1:8787" } },
  build: { rollupOptions: { input: { main: "index.html", lab: "lab.html" } } },
});
