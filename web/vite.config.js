// Relative base: the built site works from any folder, e.g. https://fangyu-qu.github.io/SteerEats/ on GitHub
// Pages, a Netlify root, or a local static server. Data files are fetched relative to the page (src/main.js).
import { defineConfig } from "vite";

export default defineConfig({ base: "./" });
