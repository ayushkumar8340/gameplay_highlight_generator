from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import time


@dataclass(frozen=True)
class RuleEvent:
    frame_idx: int
    t_sec: float
    timecode: str
    type: str                   
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RulesConfig:
    max_kill_jump: int = 3
    min_conf_accept: float = 0.45
    reset_if_zero_after_s: float = 30.0

    award_words: tuple = ("streak", "treak", "armor", "armor breaker", "elemin", "eleminator","ele","dominator","dom","do","SniperMastery")


class HUDRulesEngine:
    def __init__(self, cfg: Optional[RulesConfig] = None) -> None:
        self.cfg = cfg or RulesConfig()
        self.last_good_kills: Optional[int] = None
        self.last_accept_t: Optional[float] = None  # seconds timeline of the video

    @staticmethod
    def _extract_first_int(s: str) -> Optional[int]:
        m = re.search(r"\d+", s or "")
        return int(m.group(0)) if m else None

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
           
            if kills_int_read is not None:
                false_kill_event = True

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

        txt = (awards_text or "").strip().lower()
        if txt in self.cfg.award_words:
            events.append(RuleEvent(
                frame_idx=frame_idx, t_sec=t_sec, timecode=timecode,
                type="award_word_detected",
                payload={"word": txt},
            ))

        return events
