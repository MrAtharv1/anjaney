export type TriageDecision = "LOW_RISK_TREND" | "ELEVATED_TREND" | "ESCALATE_TO_HUMAN";

export interface TimingMetrics {
  durationSeconds: number;
  pauseCount: number;
  meanPauseDurationSeconds: number;
  typingBurstSpeedWpm: number;
}

export interface ModelBundle {
  version: string;
  encoder_name: string;
  sha256_onnx: string;
  feature_names: string[];
  scaler_mean: number[];
  scaler_scale: number[];
  ridge_coefficients: number[];
  ridge_intercept: number;
  conformal_q_hat: number;
  threshold: number;
  max_interval_width: number;
  margin: number;
}

export interface TriageResult {
  decision: TriageDecision;
  pointEstimate: number;
  interval: [number, number];
  isCrisis: boolean;
  explanation: {
    topFactor: string;
    plainText: string;
  };
}