"""Historical exp2 training reward; the current demo uses the built-in reward."""

import re
from difflib import SequenceMatcher


async def raw_reversal_score(task, trace) -> float:
    response = trace.last_reply or ""
    tagged = re.search(r"<reversed_text>(.*?)</reversed_text>", response, re.DOTALL)
    if tagged is not None:
        response = tagged.group(1)
    return SequenceMatcher(None, response.strip(), task.answer).ratio()


async def exact_reversal_score(task, trace) -> float:
    response = trace.last_reply or ""
    tagged = re.search(r"<reversed_text>(.*?)</reversed_text>", response, re.DOTALL)
    if tagged is not None:
        response = tagged.group(1)
    return float(response.strip() == task.answer)
