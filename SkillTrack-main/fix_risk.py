import json

path = 'burnout_predictions.json'
data = json.load(open(path, encoding='utf-8'))

high = med = low = 0

for s in data['students']:
    mood  = s.get('avg_mood_7d', 10)
    slope = s.get('mood_slope_7d', 0)
    conf  = s.get('confidence_score', 0)

    # HIGH — clearly burning out right now
    if mood < 3.0 or (mood < 3.8 and slope < -0.15):
        s['risk_level']      = 'HIGH'
        s['alert_counselor'] = True
        high += 1

    # MEDIUM — warning signs present
    elif mood < 5.0 or slope < -0.10 or conf >= 0.55:
        s['risk_level']      = 'MEDIUM'
        s['alert_counselor'] = True
        med += 1

    # LOW — stable
    else:
        s['risk_level']      = 'LOW'
        s['alert_counselor'] = False
        low += 1

print(f"HIGH: {high}  MEDIUM: {med}  LOW: {low}")

# Print table
print(f"\n{'ID':>4}  {'Name':<12}  {'Risk':<8}  {'Mood':>5}  {'Slope':>7}  {'Conf':>6}")
print("-" * 52)
for s in sorted(data['students'], key=lambda x: x['confidence_score'], reverse=True):
    flag = "⚠ " if s['risk_level']=='HIGH' else ("!  " if s['risk_level']=='MEDIUM' else "   ")
    print(f"{flag}{s['student_id']:>3}  {s['name']:<12}  {s['risk_level']:<8}  "
          f"{s.get('avg_mood_7d',0):>5.2f}  {s.get('mood_slope_7d',0):>7.3f}  "
          f"{s.get('confidence_score',0)*100:>5.1f}%")

# Save — no NaN allowed so PHP can read it
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

print("\nSaved! Refresh your dashboard now.")