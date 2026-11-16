import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Learn a Topic",
  description: "A study workspace for lessons, practice, and research.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
