"""Grade an answer about the current time: does it state the right time?

The right answer changes every minute, so the grader works it out when it
runs, and accepts any time within a minute or two of it. It understands both
"16:05" and "4:05 PM", so the model may use either format, but it must get
AM and PM right: "12:06 PM" for six minutes past midnight fails.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from ai_playground.evals import Case, Grade

# How far off a time may be, in minutes. The model's answer and the grader
# run a few seconds apart, so the minute can tick over in between.
TOLERANCE_MINUTES = 2

# Matches "16:05", "16:05:09", "4:05 PM", "4:05pm", "4:05 p.m." and so on.
TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?(?:\s*([AaPp])\.?\s*[Mm]\b\.?)?")


def times_in(text: str) -> list[int]:
    """Every time of day mentioned in the text, as minutes after midnight."""
    minutes = []
    for hour_text, minute_text, am_pm in TIME_PATTERN.findall(text):
        hour, minute = int(hour_text), int(minute_text)
        if am_pm:
            # 12 AM is midnight (hour 0) and 12 PM is noon (hour 12).
            hour = hour % 12 + (12 if am_pm.lower() == "p" else 0)
        if hour < 24 and minute < 60:
            minutes.append(hour * 60 + minute)
    return minutes


def now_in(timezone: str) -> int:
    """The current time in a timezone ("local" for this computer), as minutes after midnight."""
    now = datetime.now() if timezone == "local" else datetime.now(ZoneInfo(timezone))
    return now.hour * 60 + now.minute


def close_enough(a: int, b: int) -> bool:
    """Whether two times are within the tolerance, allowing for midnight in between."""
    difference = abs(a - b)
    return min(difference, 24 * 60 - difference) <= TOLERANCE_MINUTES


def as_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def states_correct_times(case: Case, answer: str) -> Grade:
    """Pass if the answer states the correct current time for every expected timezone."""
    stated = times_in(answer)
    missing = []
    for timezone in case.expected["timezones"]:
        correct = now_in(timezone)
        if not any(close_enough(time, correct) for time in stated):
            missing.append(f"{timezone} should be about {as_clock(correct)}")

    if missing:
        found = ", ".join(as_clock(time) for time in stated) or "no times"
        return Grade(False, f"{'; '.join(missing)}, but the answer has {found}")
    return Grade(True, "every expected time is correct")
