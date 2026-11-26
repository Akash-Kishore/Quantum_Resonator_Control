import json
with open(r'C:\Projects\MiniProject_Sem4_QSTUDY_WORK\results\training_health_report.json', 'r') as f:
    data = json.load(f)
for g in data['groups']:
    print(f"Group: {g['group']}, Config: {g['cfg']}, Seeds: {g['seeds']}")