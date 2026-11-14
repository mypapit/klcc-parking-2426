"""Display labels for the injected fault types.

The fault codes F1 to F6 are the identifiers used in the data, the CSV files and
the metrics. They are not always the best description of what the fault is.

F4 in particular is a single-sample excursion, not a mid-range spike: the
handover defines it as a 5-sigma departure from the value at the injection
point, and on this feed the 5-sigma magnitude is about 4,966 bays, so the
spike always leaves the valid reading range and is clamped to 0 or 5300.

These labels are for display only. Nothing here changes the data.
"""
from __future__ import annotations

FAULT_LABELS: dict[str, str] = {
    "F1": "F1 stuck sensor",
    "F2": "F2 zero block",
    "F3": "F3 full-capacity block",
    "F4": "F4 single-sample excursion (5 sigma, clamped)",
    "F5": "F5 drift",
    "F6": "F6 noise burst",
}

# the short form, for axes and legends where the long label will not fit
FAULT_SHORT: dict[str, str] = {
    "F1": "F1 stuck",
    "F2": "F2 zero block",
    "F3": "F3 full-cap",
    "F4": "F4 excursion",
    "F5": "F5 drift",
    "F6": "F6 noise",
}


def fault_label(fault: str) -> str:
    return FAULT_LABELS.get(fault, fault)


def fault_short(fault: str) -> str:
    return FAULT_SHORT.get(fault, fault)
