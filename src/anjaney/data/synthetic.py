"""Synthetic cohort generation conforming to the DAIC-WOZ schema for leak-free tests."""

import csv
import random
from pathlib import Path

from anjaney.data.schemas import Session, SpeakerRole, Turn
from anjaney.utils.reproducibility import set_seed

LOW_RISK_TEMPLATES = [
    "I had a productive study session with my friends at the library.",
    "Going for a morning run helps clear my head and gives me energy.",
    "Classes are demanding this semester, but I am keeping up fine.",
    "I talked with my roommates earlier and we made dinner together.",
    "Looking forward to the weekend break and getting some rest.",
    "I felt good about the assignment we handed in this afternoon.",
]

HIGH_RISK_TEMPLATES = [
    "I just feel completely exhausted and drained no matter how long I sleep.",
    "Nothing feels engaging or meaningful lately, just going through motions.",
    "I struggled to get out of bed this morning and skipped morning lecture.",
    "I have been avoiding messages from my family and staying by myself.",
    "Everything feels like an overwhelming uphill battle right now.",
    "I feel like I am letting everyone down constantly and cannot focus.",
]

# Structured into the 4 canonical DAIC-WOZ script phases for prompt_phase conditioning:
# Phase 0: Rapport | Phase 1: Behavior/Routine | Phase 2: Distress Probe | Phase 3: Closing
SCRIPT_PHASE_PROMPTS: dict[int, list[str]] = {
    0: [
        "rapport_intro: Where are you from originally and how is campus life?",
        "rapport_intro: Tell me a little bit about yourself and your major.",
    ],
    1: [
        "behavior_routine: Tell me about how your daily academic routine and sleep have been going.",
        "behavior_routine: What do you typically do to relax on the weekend with friends?",
    ],
    2: [
        "distress_probe: Have you been feeling down, depressed, or hopeless over the past couple of weeks?",
        "distress_probe: How have you been coping when things feel overwhelming or stressful in the past?",
    ],
    3: [
        "closing_reflection: What are your plans and goals for the rest of the semester?",
        "closing_reflection: Thank you for sharing today, is there anything else on your mind?",
    ],
}


class SyntheticCohortGenerator:
    """Generates synthetic participant sessions with statistical properties for screening."""

    def __init__(
        self,
        n_participants: int = 50,
        seed: int = 42,
        extra_high_risk_drift: bool = True,
    ):
        self.n_participants = n_participants
        self.seed = seed
        self.extra_high_risk_drift = extra_high_risk_drift

    def generate(self) -> list[Session]:
        set_seed(self.seed)
        sessions: list[Session] = []

        splits = (
            ["train"] * int(0.6 * self.n_participants)
            + ["dev"] * int(0.2 * self.n_participants)
            + ["test"] * (self.n_participants - int(0.8 * self.n_participants))
        )
        random.shuffle(splits)

        for i in range(self.n_participants):
            pid = f"SYNTH_{1000 + i}"
            # 35% baseline prevalence for elevated symptoms
            is_high_risk = random.random() < 0.35
            if is_high_risk:
                phq8_score = float(random.randint(10, 22))
                phq8_binary = 1
            else:
                phq8_score = float(random.randint(0, 9))
                phq8_binary = 0

            gender = random.choice(["F", "M"])
            split = splits[i]

            n_turns = random.randint(18, 28)
            turns: list[Turn] = []
            current_time = 2.0

            for t_idx in range(n_turns):
                position_ratio = t_idx / max(1, n_turns - 1)
                phase_id = min(3, int(position_ratio * 4))

                # 1. Interviewer prompt (used ONLY for script-phase conditioning, never embedded)
                phase_prompts = SCRIPT_PHASE_PROMPTS[phase_id]
                prompt_text = phase_prompts[t_idx % len(phase_prompts)]
                prompt_duration = round(random.uniform(2.5, 4.5), 2)
                turns.append(
                    Turn(
                        speaker=SpeakerRole.INTERVIEWER,
                        start_s=round(current_time, 2),
                        end_s=round(current_time + prompt_duration, 2),
                        text=prompt_text,
                    )
                )
                current_time += prompt_duration

                # 2. Response latency (with shared progression fatigue + optional high-risk drift)
                base_latency = 1.2 if not is_high_risk else 2.8
                fatigue_latency = position_ratio * 0.8
                extra_drift = (
                    (position_ratio * 1.5) if (is_high_risk and self.extra_high_risk_drift) else 0.0
                )

                latency = max(
                    0.4,
                    base_latency + fatigue_latency + extra_drift + random.gauss(0, 0.3),
                )
                current_time += latency

                # 3. Participant response
                use_high_template = (
                    random.random() < 0.80 if is_high_risk else random.random() < 0.15
                )
                response_text = (
                    random.choice(HIGH_RISK_TEMPLATES)
                    if use_high_template
                    else random.choice(LOW_RISK_TEMPLATES)
                )

                # Duration proportional to length with individual variation
                words = len(response_text.split())
                resp_duration = round(words / random.uniform(2.4, 3.2), 2)

                turns.append(
                    Turn(
                        speaker=SpeakerRole.PARTICIPANT,
                        start_s=round(current_time, 2),
                        end_s=round(current_time + resp_duration, 2),
                        text=response_text,
                    )
                )
                current_time += resp_duration + round(random.uniform(0.3, 0.8), 2)

            sessions.append(
                Session(
                    participant_id=pid,
                    turns=turns,
                    phq8_score=phq8_score,
                    phq8_binary=phq8_binary,
                    gender=gender,
                    official_split=split,
                    is_synthetic=True,
                )
            )

        return sessions

    def export_to_directory(self, target_dir: Path | str) -> Path:
        out_path = Path(target_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        sessions = self.generate()

        # Export labels.csv
        labels_file = out_path / "synthetic_labels.csv"
        with open(labels_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Participant_ID",
                    "PHQ8_Score",
                    "PHQ8_Binary",
                    "Gender",
                    "official_split",
                ]
            )
            for s in sessions:
                writer.writerow(
                    [
                        s.participant_id,
                        s.phq8_score,
                        s.phq8_binary,
                        s.gender,
                        s.official_split,
                    ]
                )

        # Export per-participant transcript files
        for s in sessions:
            transcript_file = out_path / f"{s.participant_id}_TRANSCRIPT.csv"
            with open(transcript_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["start_time", "stop_time", "speaker", "value"])
                for t in s.turns:
                    writer.writerow([t.start_s, t.end_s, t.speaker.value, t.text])

        return out_path
