import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.DROPIT_API_TARGET || "http://127.0.0.1:8765";
const webPort = Number.parseInt(process.env.DROPIT_WEB_PORT || "5173", 10);

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "backend/dist",
    emptyOutDir: true,
  },
  server: {
    port: webPort,
    strictPort: true,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
});
