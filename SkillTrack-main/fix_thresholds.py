import json, os

path = 'burnout_predictions.json'
data = json.load(open(path, encoding='utf-8'))

print(f"Total students: {len(data['students'])}")
print(f"Current threshold: {data['threshold']}")

# Show current distribution
from collections import Counter
risks = Counter(s['risk_level'] for s in data['students'])
print(f"Current risk distribution: {dict(risks)}")
print()

# Show confidence score range
confs = sorted([s['confidence_score'] for s in data['students']], reverse=True)
print(f"Confidence scores (top 15): {[round(c*100,1) for c in confs[:15]]}")
print(f"Confidence scores (bottom 8): {[round(c*100,1) for c in confs[-8:]]}")
print()

# Re-label with corrected thresholds based on actual data patterns
# HIGH:   conf >= 0.60 AND (avg_mood_7d < 3.5 OR mood_slope_7d < -0.15)
# MEDIUM: conf >= 0.50 OR avg_mood_7d < 4.5
# LOW:    everything else

relabeled = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
for s in data['students']:
    conf  = s['confidence_score']
    mood  = s.get('avg_mood_7d', 10)
    slope = s.get('mood_slope_7d', 0)

    if conf >= 0.60 and (mood < 3.5 or slope < -0.15):
        s['risk_level'] = 'HIGH'
        s['alert_counselor'] = True
    elif conf >= 0.52 or mood < 4.5:
        s['risk_level'] = 'MEDIUM'
        s['alert_counselor'] = True
    else:
        s['risk_level'] = 'LOW'
        s['alert_counselor'] = False

    relabeled[s['risk_level']] += 1

print(f"New risk distribution: {relabeled}")

# Save
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

print()
print("=== UPDATED RISK REPORT ===")
print(f"{'ID':>4}  {'Name':<12}  {'Risk':<8}  {'Conf':>6}  {'Mood':>5}  {'Slope':>7}")
print("-" * 55)
for s in data['students']:
    flag = "⚠" if s['risk_level'] == 'HIGH' else ("!" if s['risk_level'] == 'MEDIUM' else " ")
    print(f"{flag} {s['student_id']:>3}  {s['name']:<12}  {s['risk_level']:<8}  "
          f"{s['confidence_score']*100:>5.1f}%  {s.get('avg_mood_7d',0):>5.2f}  "
          f"{s.get('mood_slope_7d',0):>7.3f}")

print()
print(f"Fixed! HIGH={relabeled['HIGH']}  MEDIUM={relabeled['MEDIUM']}  LOW={relabeled['LOW']}")
