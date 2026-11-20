"""Choose a cushion multiplier from completed monthly starts."""

from statistics import fmean, harmonic_mean

MULTIPLIERS = (2, 4, 6, 8, 12)
HYBRID = "cushion-capped-dp-combined"


def _completed_stats(rows, cutoff, conditions):
    past = [row for row in rows if row.last_session < cutoff]
    wealth = [row.terminal_wealth for row in past]
    stats = dict(selection_date=cutoff, latest_completion=max((r.last_session for r in past), default=None),
                 ce=harmonic_mean(wealth) if wealth else 0, mean_wealth=fmean(wealth) if wealth else 0)
    for condition in conditions:
        group = [row for row in past if row.condition == condition]
        breaches = sum(row.floor_breach for row in group)
        stats[condition + "_count"] = len(group)
        stats[condition + "_breaches"] = breaches
        stats[condition + "_rate"] = breaches / len(group) if group else None
    return stats


def select_multiplier(history, cutoff, conditions, breach_limit):
    """Score only outcomes observable before the first session of the year."""
    audit = []
    selected = None
    best_ce = best_mean = 0
    for multiplier in MULTIPLIERS:
        record = _completed_stats(history[multiplier], cutoff, conditions)
        record["multiplier"] = multiplier
        ce, mean = record["ce"], record["mean_wealth"]
        qualifies = all(record[c + "_count"] >= 50 and record[c + "_breaches"] <= breach_limit * record[c + "_count"]
                        for c in conditions)
        record["qualifies"] = qualifies
        audit.append(record)
        if qualifies and (selected is None or (ce > best_ce and mean > best_mean)):
            selected, best_ce, best_mean = multiplier, ce, mean

    reason = "qualified"
    if selected is None:
        selected = MULTIPLIERS[0]
        reason = ("insufficient history" if any(audit[0][c + "_count"] < 50 for c in conditions)
                  else "no qualifying multiplier")
    for record in audit:
        record.update(selected_multiplier=selected, reason=reason)
    return selected, audit


TIME_AWARE = "time-aware-smooth-cushion"
SMOOTH_SCALES = (0., .5, .75, 1., 1.25, 1.5)


def select_smooth_scale(history, cutoff, conditions, breach_limit):
    """None selects the nominal smooth policy until all launch groups have coverage."""
    audit = []
    selected = None
    best_mean = float("-inf")
    for scale in SMOOTH_SCALES:
        row = _completed_stats(history[scale], cutoff, conditions)
        row["scale"] = scale
        qualifies = all(row[c + "_count"] >= 50 and row[c + "_breaches"] < breach_limit * row[c + "_count"]
                        for c in conditions)
        row["qualifies"] = qualifies
        if qualifies and row["mean_wealth"] > best_mean:
            best_mean, selected = row["mean_wealth"], scale
        audit.append(row)
    if any(audit[0][group + "_count"] < 50 for group in conditions):
        mode = "nominal warm-up"
    elif selected is None:
        mode, selected = "cash fallback", 0.
    else:
        mode = "calibrated"
    for row in audit:
        row.update(selected_scale=selected, mode=mode, selected=row["scale"] == selected)
    return selected, audit
