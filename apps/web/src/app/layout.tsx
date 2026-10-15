import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ANJANEY Campus Wellness",
  description: "Personal-baseline, zero-leakage campus screening aid",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}