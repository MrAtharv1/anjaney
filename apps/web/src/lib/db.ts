import Dexie, { type Table } from "dexie";
import { TriageDecision } from "./types";

export interface StoredCheckin {
  id?: number;
  timestamp: string;
  decision: TriageDecision;
  intervalLower: number;
  intervalUpper: number;
  pointEstimate: number;
  baselineDeviation: number;
  // Raw text is NEVER persisted by default
}

export interface StoredBaseline {
  id: string; // 'current_baseline'
  centroid: number[];
  checkinCount: number;
  updatedAt: string;
}

export class AnjaneyDatabase extends Dexie {
  checkins!: Table<StoredCheckin>;
  baselines!: Table<StoredBaseline>;

  constructor() {
    super("AnjaneyLocalDB");
    this.version(1).stores({
      checkins: "++id, timestamp, decision",
      baselines: "id, updatedAt",
    });
  }
}

export const db = new AnjaneyDatabase();