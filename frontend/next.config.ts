import type { NextConfig } from "next";

const config: NextConfig = {
  devIndicators: false,
  // Approval can run both the Tutor and Assessment before the API responds.
  experimental: { proxyTimeout: 190_000 },
  async rewrites() {
    return [{
      source: "/api/study/:action",
      destination: "http://127.0.0.1:8787/:action",
    }];
  },
};

export default config;
