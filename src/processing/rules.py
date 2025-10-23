# --- src/processing/rules.py ---
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import time


@dataclass(frozen=True)
class RuleEvent:
    """One key moment emitted by the rules engine."""
    frame_idx: int
    t_sec: float
    timecode: str
    type: str                   # 'kill_while_low_health' | 'false_kill_while_low_health' | 'award_word_detected'
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RulesConfig:
    # kill acceptance
    max_kill_jump: int = 3
    min_conf_accept: float = 0.45
    reset_if_zero_after_s: float = 30.0

    # award words (case-insensitive exact match)
    award_words: tuple = ("streak", "treak", "armor", "armor breaker")


class HUDRulesEngine:
    """
    Pure rules engine that:
      - stabilizes kill count from noisy OCR
      - flags 'kill while low health' and 'false kill while low health'
      - flags exact award word hits (case-insensitive) among {streak, treak, armor, armor breaker}

    Public API: process(...) -> List[RuleEvent]
    State kept internally: last accepted kill count + last accept time.
    """

    def __init__(self, cfg: Optional[RulesConfig] = None) -> None:
        self.cfg = cfg or RulesConfig()
        self.last_good_kills: Optional[int] = None
        self.last_accept_t: Optional[float] = None  # seconds timeline of the video

    # ---------- helpers ----------
    @staticmethod
    def _extract_first_int(s: str) -> Optional[int]:
        m = re.search(r"\d+", s or "")
        return int(m.group(0)) if m else None

    # ---------- core ----------
    def process(
        self,
        *,
        frame_idx: int,
        t_sec: float,
        timecode: str,
        kills_raw_text: str,
        kills_avg_conf: float,
        health_state: str,
        health_low: bool,
        health_severity: float,
        awards_text: str,
    ) -> List[RuleEvent]:
        """
        Feed one frame worth of measurements. Returns a list of RuleEvent emitted for this frame.
        - kills_raw_text: raw OCR text from kill counter region
        - kills_avg_conf: average confidence across spans for kill OCR
        - health_state: detector's state string (e.g., "low_health" or "ok")
        - health_low: True if low health
        - health_severity: numeric severity (0..100 if you use that scale)
        - awards_text: raw OCR text from the HUD awards region
        """
        events: List[RuleEvent] = []

        # ----- Kills acceptance (stabilization) -----
        kills_int_read = self._extract_first_int(kills_raw_text)
        accept = False
        kill_event = False
        false_kill_event = False

        if kills_int_read is not None:
            if kills_avg_conf >= self.cfg.min_conf_accept:
                if self.last_good_kills is None:
                    accept = True
                else:
                    diff = kills_int_read - self.last_good_kills
                    if 0 <= diff <= self.cfg.max_kill_jump:
                        accept = True
                    elif kills_int_read == 0 and (self.last_good_kills or 0) > 0:
                        if self.last_accept_t is None or (t_sec - self.last_accept_t) >= self.cfg.reset_if_zero_after_s:
                            accept = True

        if accept:
            prev = self.last_good_kills
            self.last_good_kills = kills_int_read
            self.last_accept_t = t_sec
            if prev is not None and kills_int_read is not None and kills_int_read > prev:
                kill_event = True
        else:
            # do not increment accepted kills; still treat as a possible kill for correlation
            if kills_int_read is not None:
                false_kill_event = True

        # ----- Correlate with health -----
        if kill_event and health_low:
            events.append(RuleEvent(
                frame_idx=frame_idx, t_sec=t_sec, timecode=timecode,
                type="kill_while_low_health",
                payload={
                    "kills_accepted": self.last_good_kills,
                    "health_state": health_state,
                    "health_severity": health_severity
                },
            ))

        if false_kill_event and health_low:
            events.append(RuleEvent(
                frame_idx=frame_idx, t_sec=t_sec, timecode=timecode,
                type="false_kill_while_low_health",
                payload={
                    "health_state": health_state,
                    "health_severity": health_severity
                },
            ))

        # ----- Award word detection (exact, case-insensitive) -----
        txt = (awards_text or "").strip().lower()
        if txt in self.cfg.award_words:
            events.append(RuleEvent(
                frame_idx=frame_idx, t_sec=t_sec, timecode=timecode,
                type="award_word_detected",
                payload={"word": txt},
            ))

        return events
