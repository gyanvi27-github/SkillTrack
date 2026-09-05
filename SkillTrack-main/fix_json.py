import json, re

path = 'burnout_predictions.json'

raw = open(path, encoding='utf-8').read()
print("Before - size:", len(raw))

# Replace NaN and Infinity which PHP cannot parse
raw = raw.replace(': NaN', ': null')
raw = raw.replace(':NaN',  ':null')
raw = raw.replace(': Infinity', ': null')
raw = raw.replace(': -Infinity', ': null')

# Verify still valid Python JSON
data = json.loads(raw)
print("Students:", len(data['students']))

# Save fixed version
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

print("Fixed! Size:", len(open(path, encoding='utf-8').read()))
print("Checking for NaN:", 'NaN' in open(path, encoding='utf-8').read())