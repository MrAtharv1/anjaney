"use client";

import React, { useState, useRef, useEffect } from "react";
import { evaluateModelPrediction, extractClientLexicalFeatures } from "../lib/inference";
import { ModelBundle, TriageResult } from "../lib/types";
import { db } from "../lib/db";
import { ShieldCheck, HeartHandshake, Sparkles, RefreshCw, AlertCircle } from "lucide-react";

interface Props {
  bundle: ModelBundle;
}

export function CheckinScreen({ bundle }: Props) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<TriageResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [totalCheckins, setTotalCheckins] = useState(0);

  const startTime = useRef<number | null>(null);
  const pauseCount = useRef<number>(0);
  const lastKeyTime = useRef<number | null>(null);

  useEffect(() => {
    db.checkins.count().then(setTotalCheckins);
  }, []);

  const handleKeyDown = () => {
    const now = performance.now();
    if (!startTime.current) startTime.current = now;

    if (lastKeyTime.current && (now - lastKeyTime.current) > 1500) {
      pauseCount.current += 1;
    }
    lastKeyTime.current = now;
  };

  const handleScoreCheckin = async () => {
    setIsSubmitting(true);
    const now = performance.now();
    const duration = startTime.current ? (now - startTime.current) / 1000 : 8.0;

    // 1. Deterministic Crisis Keyword Guard (tele-manas override)
    const crisisMatch = text.match(/\b(suicide|kill myself|end my life|want to die|cut myself|jaan deni)\b/i);
    if (crisisMatch) {
      setResult({
        decision: "ESCALATE_TO_HUMAN",
        pointEstimate: 20.0,
        interval: [16.0, 24.0],
        isCrisis: true,
        explanation: {
          topFactor: "Immediate Safety Trigger",
          plainText: "We noticed words indicating severe distress. Your safety is what matters most right now.",
        },
      });
      setIsSubmitting(false);
      return;
    }

    // 2. Extract genuine lexical metrics
    const lex = extractClientLexicalFeatures(text);

    // 3. Build model vector matching bundle coordinates
    const inputVector = new Array(bundle.feature_names.length).fill(0.0);
    
    // Populate known feature positions
    bundle.feature_names.forEach((name, idx) => {
      if (name.includes("mean_turn_duration") || name.includes("duration")) {
        inputVector[idx] = duration;
      } else if (name.includes("mean_latency") || name.includes("pause")) {
        inputVector[idx] = pauseCount.current * 0.8;
      } else if (name.includes("first_person")) {
        inputVector[idx] = lex.firstPersonRatio;
      } else if (name.includes("vader_compound") || name.includes("neg")) {
        inputVector[idx] = lex.negativeAffect;
      } else if (name.includes("word_count")) {
        inputVector[idx] = lex.wordCount;
      } else {
        // Fallback to scaler training mean so unmeasured embeddings sit at zero z-score
        inputVector[idx] = bundle.scaler_mean[idx] ?? 0.0;
      }
    });

    const nextCount = totalCheckins + 1;
    const triage = evaluateModelPrediction(inputVector, bundle, nextCount);

    // Persist to local IndexedDB (zero raw text stored)
    await db.checkins.add({
      timestamp: new Date().toISOString(),
      decision: triage.decision,
      intervalLower: triage.interval[0],
      intervalUpper: triage.interval[1],
      pointEstimate: triage.pointEstimate,
      baselineDeviation: 0.12,
    });

    setTotalCheckins(nextCount);
    setResult(triage);
    setIsSubmitting(false);
  };

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-500" />
            ANJANEY Campus Wellness
          </h1>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Personal-baseline, zero-leakage screening aid
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-[11px] font-medium text-emerald-700 dark:text-emerald-300">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>On-Device Inference</span>
        </div>
      </div>

      {!result ? (
        <div className="bg-white dark:bg-zinc-900 rounded-2xl p-6 shadow-sm border border-zinc-200/80 dark:border-zinc-800 space-y-4">
          <div className="flex justify-between items-baseline">
            <label className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
              How has your day felt overall?
            </label>
            <span className="text-[11px] text-zinc-400">
              {totalCheckins < 3
                ? `Baseline calibration: ${totalCheckins}/3 check-ins`
                : `${totalCheckins} check-ins logged`}
            </span>
          </div>

          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Write casually about your routine, sleep, academics, or mood. Nothing you type leaves this browser..."
            className="w-full p-4 rounded-xl border border-zinc-200 dark:border-zinc-700/80 bg-zinc-50/50 dark:bg-zinc-800/40 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 transition resize-none"
          />

          <div className="flex items-center justify-between pt-1">
            <span className="text-xs text-zinc-400">
              {wordCount} words {wordCount < 4 && "(write at least a few words)"}
            </span>
            <button
              onClick={handleScoreCheckin}
              disabled={isSubmitting || wordCount < 3}
              className="px-5 py-2.5 rounded-xl bg-zinc-900 dark:bg-zinc-100 hover:bg-zinc-800 dark:hover:bg-zinc-200 text-white dark:text-zinc-900 text-xs font-semibold tracking-wide transition shadow-sm disabled:opacity-40"
            >
              {isSubmitting ? "Evaluating Locally..." : "Submit Check-In"}
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Result Card */}
          <div
            className={`rounded-2xl p-6 border shadow-sm ${
              result.isCrisis
                ? "bg-rose-50 border-rose-300 dark:bg-rose-950/40 dark:border-rose-800"
                : result.decision === "LOW_RISK_TREND"
                ? "bg-emerald-50/60 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900/60"
                : result.decision === "ELEVATED_TREND"
                ? "bg-amber-50/60 border-amber-200 dark:bg-amber-950/30 dark:border-amber-900/60"
                : "bg-blue-50/60 border-blue-200 dark:bg-blue-950/30 dark:border-blue-900/60"
            }`}
          >
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  {result.isCrisis ? "Emergency Safeguard" : "Check-In Insight"}
                </span>
                <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 mt-1">
                  {result.isCrisis
                    ? "Immediate Support Recommended"
                    : result.decision === "LOW_RISK_TREND"
                    ? "Usual-Range Pattern"
                    : result.decision === "ELEVATED_TREND"
                    ? "Noticeable Pattern Shift"
                    : "Peer Counselor Check-In Suggested"}
                </h3>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-[11px] font-semibold ${
                  result.decision === "LOW_RISK_TREND"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300"
                    : result.decision === "ELEVATED_TREND"
                    ? "bg-amber-100 text-amber-800 dark:bg-amber-900/60 dark:text-amber-300"
                    : "bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-300"
                }`}
              >
                {result.decision.replace(/_/g, " ")}
              </span>
            </div>

            <p className="text-sm text-zinc-600 dark:text-zinc-300 mt-3 leading-relaxed">
              {result.explanation.plainText}
            </p>

            {/* Score Range Visualization */}
            {!result.isCrisis && (
              <div className="mt-5 pt-4 border-t border-zinc-200/60 dark:border-zinc-800/80 space-y-2">
                <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400 font-mono">
                  <span>Lower: {result.interval[0].toFixed(1)}</span>
                  <span className="font-bold text-zinc-800 dark:text-zinc-200">
                    PHQ-8 Score: {result.pointEstimate.toFixed(1)} / 24
                  </span>
                  <span>Upper: {result.interval[1].toFixed(1)}</span>
                </div>

                <div className="relative w-full h-3 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 bottom-0 bg-emerald-500/70 rounded-full transition-all"
                    style={{
                      left: `${(result.interval[0] / 24) * 100}%`,
                      width: `${Math.max(4, ((result.interval[1] - result.interval[0]) / 24) * 100)}%`,
                    }}
                  />
                  {/* Threshold Line at 10.0 */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
                    style={{ left: `${(10.0 / 24) * 100}%` }}
                    title="Threshold: 10.0"
                  />
                </div>
                <div className="flex justify-between text-[10px] text-zinc-400">
                  <span>0 (Minimal)</span>
                  <span className="text-rose-500 font-semibold">10.0 Triage Cutoff</span>
                  <span>24 (Severe)</span>
                </div>
              </div>
            )}
          </div>

          {/* Crisis / Support Card */}
          <div className="p-4 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2.5">
              <HeartHandshake className="w-4 h-4 text-rose-500 flex-shrink-0" />
              <span className="text-zinc-600 dark:text-zinc-300">
                Government Mental Health Helpline (24x7):
              </span>
            </div>
            <a
              href="tel:14416"
              className="font-bold text-emerald-600 dark:text-emerald-400 hover:underline"
            >
              Tele-MANAS: 14416
            </a>
          </div>

          <button
            onClick={() => {
              setResult(null);
              setText("");
              startTime.current = null;
              pauseCount.current = 0;
            }}
            className="w-full py-2.5 rounded-xl border border-zinc-300 dark:border-zinc-700 text-xs font-semibold text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Start Another Check-In
          </button>
        </div>
      )}
    </div>
  );
}