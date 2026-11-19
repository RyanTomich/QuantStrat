"""Choose a cushion multiplier from completed monthly starts."""

from statistics import fmean, harmonic_mean

MULTIPLIERS = (2, 4, 6, 8, 12)
HYBRID = "cushion-capped-dp-combined"


def select_multiplier(history, cutoff, conditions, breach_limit):
    """Score only outcomes observable before the first session of the year."""
    audit = []
    selected = None
    best_ce = best_mean = 0
    for multiplier in MULTIPLIERS:
        past = [row for row in history[multiplier] if row.last_session < cutoff]
        wealth = [row.terminal_wealth for row in past]
        ce = harmonic_mean(wealth) if wealth else 0
        mean = fmean(wealth) if wealth else 0
        record = dict(selection_date=cutoff, multiplier=multiplier,
                      latest_completion=max((r.last_session for r in past), default=None),
                      ce=ce, mean_wealth=mean)
        qualifies = True
        for condition in conditions:
            group = [r for r in past if r.condition == condition]
            breaches = sum(r.floor_breach for r in group)
            record[condition + "_count"] = len(group)
            record[condition + "_breaches"] = breaches
            record[condition + "_rate"] = breaches / len(group) if group else None
            qualifies &= len(group) >= 50 and breaches <= breach_limit * len(group)
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
