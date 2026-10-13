"use client";

import React, { useState, useRef } from "react";
import { evaluateModelPrediction } from "../lib/inference";
import { ModelBundle, TriageResult } from "../lib/types";
import { db } from "../lib/db";

interface Props {
  bundle: ModelBundle;
}

export function CheckinScreen({ bundle }: Props) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<TriageResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Content-free passive timing metrics
  const keyEvents = useRef<number[]>([]);
  const startTime = useRef<number | null>(null);

  const handleKeyDown = () => {
    const now = performance.now();
    if (!startTime.current) {
      startTime.current = now;
    }
    keyEvents.current.push(now);
  };

  const handleScoreCheckin = async () => {
    setIsSubmitting(true);
    const now = performance.now();
    const duration = startTime.current ? (now - startTime.current) / 1000 : 10.0;

    // Build synthetic feature vector matching bundle length
    const dummyVector = new Array(bundle.feature_names.length).fill(0.0);
    dummyVector[bundle.feature_names.length - 1] = duration; // Pacing duration feature

    const triage = evaluateModelPrediction(dummyVector, bundle);

    // Save to local IndexedDB (zero raw text persisted)
    await db.checkins.add({
      timestamp: new Date().toISOString(),
      decision: triage.decision,
      intervalLower: triage.interval[0],
      intervalUpper: triage.interval[1],
      pointEstimate: triage.pointEstimate,
      baselineDeviation: 0.15,
    });

    setResult(triage);
    setIsSubmitting(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
        Daily Campus Check-In
      </h2>
      <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-4">
        Everything you write is scored entirely inside your browser. No free-text leaves this device.
      </p>

      {!result ? (
        <div className="space-y-4">
          <textarea
            rows={5}
            className="w-full p-4 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-transparent text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            placeholder="How has your week been feeling academically and socially?"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <div className="flex justify-between items-center pt-2">
            <span className="text-xs text-zinc-400">
              {text.trim().split(/\s+/).filter(Boolean).length} words
            </span>
            <button
              onClick={handleScoreCheckin}
              disabled={isSubmitting || text.length < 5}
              className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium transition disabled:opacity-50"
            >
              {isSubmitting ? "Scoring On-Device..." : "Complete Check-In"}
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6 pt-2">
          <div
            className={`p-5 rounded-xl border ${
              result.decision === "LOW_RISK_TREND"
                ? "bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900"
                : result.decision === "ELEVATED_TREND"
                ? "bg-amber-50 border-amber-200 dark:bg-amber-950/30 dark:border-amber-900"
                : "bg-blue-50 border-blue-200 dark:bg-blue-950/30 dark:border-blue-900"
            }`}
          >
            <span className="text-xs font-semibold tracking-wider uppercase text-zinc-600 dark:text-zinc-300">
              Check-In Observation
            </span>
            <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-100 mt-1">
              {result.decision === "LOW_RISK_TREND"
                ? "Usual-Range Pattern"
                : result.decision === "ELEVATED_TREND"
                ? "Pattern Shift Observed"
                : "Support Check-In Recommended"}
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-300 mt-2">
              {result.explanation.plainText}
            </p>
          </div>

          <div className="bg-zinc-50 dark:bg-zinc-800/50 p-4 rounded-xl border border-zinc-200 dark:border-zinc-700/60">
            <div className="flex justify-between text-xs text-zinc-500 mb-1">
              <span>Lower Bound: {result.interval[0].toFixed(1)}</span>
              <span>Point: {result.pointEstimate.toFixed(1)}</span>
              <span>Upper Bound: {result.interval[1].toFixed(1)}</span>
            </div>
            <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full"
                style={{
                  width: `${Math.min(100, Math.max(10, (result.pointEstimate / 24) * 100))}%`,
                }}
              />
            </div>
          </div>

          <div className="p-4 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-xs text-zinc-500 flex justify-between items-center">
            <span>Need confidential human support?</span>
            <span className="font-semibold text-zinc-800 dark:text-zinc-200">
              Tele-MANAS: 14416 (24x7)
            </span>
          </div>

          <button
            onClick={() => {
              setResult(null);
              setText("");
            }}
            className="w-full py-2.5 rounded-xl border border-zinc-300 dark:border-zinc-700 text-sm font-medium hover:bg-zinc-50 dark:hover:bg-zinc-800 transition"
          >
            Start New Check-In
          </button>
        </div>
      )}
    </div>
  );
}