import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { evaluateModelPrediction } from "../src/lib/inference";
import { ModelBundle } from "../src/lib/types";

describe("Python <-> TypeScript Engine Parity Test", () => {
  it("matches Python reference outputs within numerical tolerance", () => {
    const fixturePath = join(__dirname, "../../../tests/fixtures/parity.json");
    const bundlePath = join(__dirname, "../../../data/processed/bundle/bundle.json");

    if (!existsSync(fixturePath) || !existsSync(bundlePath)) {
      console.warn("Parity fixtures or bundle not found. Skipping parity assertion.");
      return;
    }

    const fixtureData = JSON.parse(readFileSync(fixturePath, "utf-8"));
    const bundle: ModelBundle = JSON.parse(readFileSync(bundlePath, "utf-8"));

    const fixtures = fixtureData.parity_fixtures;
    expect(fixtures.length).toBeGreaterThan(0);

    for (const item of fixtures) {
      const res = evaluateModelPrediction(item.input_vector, bundle);

      // Verify point estimate and interval boundaries within 0.05 tolerance
      expect(Math.abs(res.pointEstimate - item.expected_point_score)).toBeLessThanOrEqual(0.05);
      expect(Math.abs(res.interval[0] - item.expected_lower)).toBeLessThanOrEqual(0.05);
      expect(Math.abs(res.interval[1] - item.expected_upper)).toBeLessThanOrEqual(0.05);
    }
  });
});