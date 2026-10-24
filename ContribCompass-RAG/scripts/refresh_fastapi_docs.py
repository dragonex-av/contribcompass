"""Refresh the real source documents; no synthetic contributor guidance."""
import json
import ssl
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import certifi

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'FASTAPI_README.md': 'https://raw.githubusercontent.com/fastapi/fastapi/master/README.md',
    'DEVELOPMENT_SETUP.md': 'https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/virtual-environments.md',
    'DEPLOYMENT.md': 'https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/deployment/manually.md',
    'COMMUNITY_GUIDE.md': 'https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/help-fastapi.md',
    'CONTRIBUTING_GUIDE.md': 'https://raw.githubusercontent.com/tiangolo/tiangolo.com/master/docs/open-source/contributing.md',
}


def main():
    context = ssl.create_default_context(cafile=certifi.where())
    # Fetch all first, so a network failure cannot leave a half-refreshed corpus.
    downloaded = {}
    for name, url in SOURCES.items():
        with urlopen(url, context=context, timeout=25) as response:
            downloaded[name] = response.read().decode('utf-8')
    for name, text in downloaded.items():
        (ROOT / 'documents' / name).write_text(text)
        print('Refreshed', name)
    (ROOT / 'documents/sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    (ROOT / 'documents/provenance.json').write_text(json.dumps({
        'verified_on': date.today().isoformat(), 'sources': SOURCES,
        'notes': 'Downloaded directly from public maintainer source files.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
