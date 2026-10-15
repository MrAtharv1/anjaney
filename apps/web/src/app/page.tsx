"use client";

import { useEffect, useState } from "react";
import { CheckinScreen } from "../components/CheckinScreen";
import { ModelBundle } from "../lib/types";

export default function Home() {
  const [bundle, setBundle] = useState<ModelBundle | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch the versioned bundle from the FastAPI backend
    fetch("http://localhost:8000/api/v1/model/bundle")
      .then((res) => {
        if (!res.ok) throw new Error("Could not load model bundle from backend.");
        return res.json();
      })
      .then((data) => setBundle(data.bundle))
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <main className="min-h-screen flex items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
        <div className="p-6 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 rounded-xl text-red-700 dark:text-red-300 text-sm max-w-md">
          <p className="font-semibold mb-1">Backend Connection Error</p>
          <p>{error}</p>
          <p className="mt-2 text-xs text-zinc-500">
            Make sure the FastAPI server is running on port 8000.
          </p>
        </div>
      </main>
    );
  }

  if (!bundle) {
    return (
      <main className="min-h-screen flex items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
        <p className="text-sm text-zinc-500 animate-pulse">Loading ANJANEY On-Device Bundle...</p>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-6 md:p-12 bg-zinc-50 dark:bg-zinc-950">
      <CheckinScreen bundle={bundle} />
    </main>
  );
}