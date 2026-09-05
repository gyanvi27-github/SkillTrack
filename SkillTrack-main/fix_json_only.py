import json

path = 'burnout_predictions.json'

# Read raw file
raw = open(path, encoding='utf-8').read()
print("Original size:", len(raw))

# Fix NaN and Infinity - PHP cannot read these
raw = raw.replace(': NaN', ': null')
raw = raw.replace(':NaN', ':null')
raw = raw.replace(': Infinity', ': null')
raw = raw.replace(': -Infinity', ': null')
raw = raw.replace(':Infinity', ':null')
raw = raw.replace(':-Infinity', ':null')

# Verify Python can parse it
data = json.loads(raw)
print("Students found:", len(data.get('students', [])))

# Write back with no NaN allowed
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

# Verify final file
final = open(path, encoding='utf-8').read()
print("NaN remaining:", 'NaN' in final)
print("Fixed size:", len(final))
print("Done! JSON is now PHP-safe.")