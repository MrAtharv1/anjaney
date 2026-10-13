/**
 * Pure client-side mathematical evaluation matching Python backend exactly.
 */
import { ModelBundle, TriageDecision, TriageResult } from "./types";

export function evaluateModelPrediction(
  features: number[],
  bundle: ModelBundle
): TriageResult {
  // 1. Standardize features: (x - mean) / scale
  const scaled = features.map((val, i) => {
    const mean = bundle.scaler_mean[i] ?? 0.0;
    const scale = bundle.scaler_scale[i] ?? 1.0;
    return (val - mean) / (scale === 0 ? 1.0 : scale);
  });

  // 2. Ridge prediction: intercept + sum(coef * scaled)
  let yPred = bundle.ridge_intercept;
  for (let i = 0; i < scaled.length; i++) {
    yPred += (bundle.ridge_coefficients[i] ?? 0.0) * scaled[i];
  }

  // 3. Conformal prediction interval
  const lower = yPred - bundle.conformal_q_hat;
  const upper = yPred + bundle.conformal_q_hat;
  const width = upper - lower;

  // 4. Decision Policy
  let decision: TriageDecision;
  const crosses = (lower - bundle.margin) <= bundle.threshold && bundle.threshold <= (upper + bundle.margin);

  if (crosses || width > bundle.max_interval_width) {
    decision = "ESCALATE_TO_HUMAN";
  } else if (lower > bundle.threshold) {
    decision = "ELEVATED_TREND";
  } else {
    decision = "LOW_RISK_TREND";
  }

  const plainText =
    decision === "LOW_RISK_TREND"
      ? "Your check-in pattern aligns comfortably with your established campus baseline."
      : decision === "ELEVATED_TREND"
      ? "Your pattern reflects notable variation from your typical baseline across pacing and expression."
      : "We noticed some pattern variation that would benefit from a quick, confidential chat with a campus counselor.";

  return {
    decision,
    pointEstimate: yPred,
    interval: [lower, upper],
    isCrisis: false,
    explanation: {
      topFactor: "Conversational Pacing & Expression",
      plainText,
    },
  };
}