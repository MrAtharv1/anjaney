import { ModelBundle, TriageDecision, TriageResult } from "./types";

export interface LexicalFeatures {
  wordCount: number;
  firstPersonRatio: number;
  absolutistRatio: number;
  typeTokenRatio: number;
  negativeAffect: number;
}

export function extractClientLexicalFeatures(text: string): LexicalFeatures {
  const tokens = text.toLowerCase().match(/\b\w+\b/g) || [];
  const total = tokens.length;
  if (total === 0) {
    return {
      wordCount: 0,
      firstPersonRatio: 0,
      absolutistRatio: 0,
      typeTokenRatio: 0,
      negativeAffect: 0,
    };
  }

  const firstPerson = new Set(["i", "me", "my", "myself", "we", "us", "our"]);
  const absolutist = new Set(["always", "never", "completely", "totally", "all", "none", "every", "nothing"]);
  const negativeWords = new Set(["bad", "exhausted", "tired", "sad", "hate", "drained", "worst", "anxious", "fail", "hopeless"]);

  let fpCount = 0;
  let absCount = 0;
  let negCount = 0;

  for (const t of tokens) {
    if (firstPerson.has(t)) fpCount++;
    if (absolutist.has(t)) absCount++;
    if (negativeWords.has(t)) negCount++;
  }

  return {
    wordCount: total,
    firstPersonRatio: fpCount / total,
    absolutistRatio: absCount / total,
    typeTokenRatio: new Set(tokens).size / total,
    negativeAffect: negCount / total,
  };
}

export function evaluateModelPrediction(
  features: number[],
  bundle: ModelBundle,
  checkinCount: number = 1
): TriageResult {
  // 1. Feature normalization
  const scaled = features.map((val, i) => {
    const mean = bundle.scaler_mean[i] ?? 0.0;
    const scale = bundle.scaler_scale[i] ?? 1.0;
    return (val - mean) / (scale === 0 ? 1.0 : scale);
  });

  // 2. Ridge prediction
  let rawY = bundle.ridge_intercept;
  for (let i = 0; i < scaled.length; i++) {
    rawY += (bundle.ridge_coefficients[i] ?? 0.0) * scaled[i];
  }

  // Clamped clinical PHQ-8 bounds [0, 24]
  const clampedPoint = Math.min(24.0, Math.max(0.0, rawY));
  const qHat = bundle.conformal_q_hat;

  const lower = Math.min(24.0, Math.max(0.0, clampedPoint - qHat));
  const upper = Math.min(24.0, Math.max(0.0, clampedPoint + qHat));
  const width = upper - lower;

  // 3. Triage Decision Policy
  let decision: TriageDecision;
  const crosses = (lower - bundle.margin) <= bundle.threshold && bundle.threshold <= (upper + bundle.margin);

  if (crosses || width > bundle.max_interval_width) {
    decision = "ESCALATE_TO_HUMAN";
  } else if (lower >= bundle.threshold) {
    decision = "ELEVATED_TREND";
  } else {
    decision = "LOW_RISK_TREND";
  }

  // Non-diagnostic explanation
  let plainText = "";
  if (checkinCount < 3) {
    plainText = `Baseline calibration check-in (${checkinCount} of 3). Your pattern will stabilize after your first 3 check-ins.`;
  } else if (decision === "LOW_RISK_TREND") {
    plainText = "Your check-in pattern aligns comfortably within your established personal campus baseline.";
  } else if (decision === "ELEVATED_TREND") {
    plainText = "Your expression shows a noticeable deviation from your typical baseline across pacing and sentiment.";
  } else {
    plainText = "We noticed some pattern variation that would benefit from an informal, confidential chat with a peer counselor.";
  }

  return {
    decision,
    pointEstimate: clampedPoint,
    interval: [lower, upper],
    isCrisis: false,
    explanation: {
      topFactor: "Linguistic Sentiment & Pacing",
      plainText,
    },
  };
}