import os
import yaml

REQUIRED_KEYS = [
    'demo_path', 'smq_path', 'rdq_path', 'mcq_path', 'ocq_path',
    'cotnal_path', 'cbc_path', 'spx_g_path', 'spxraw_g_path'
]

cfg = yaml.safe_load(open('config.yaml', 'r'))

missing = []
for key in REQUIRED_KEYS:
    path = cfg['data'].get(key, '')
    if not os.path.exists(path):
        missing.append(path)

if missing:
    print('Missing data files:')
    for m in missing:
        print(' -', m)
else:
    print('All required data files are present.')
