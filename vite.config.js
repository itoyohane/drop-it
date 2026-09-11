import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.DROPIT_API_TARGET || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
});
