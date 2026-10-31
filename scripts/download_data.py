#!/usr/bin/env python3
"""Download official inputs, or import already downloaded copies. No credentials."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/'
SOURCES = {
    'meta-New_York.json.gz': BASE + 'meta-New_York.json.gz',
    'review-New_York.json.gz': BASE + 'review-New_York.json.gz',
    'nyc_nta_2020.geojson': 'https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=5000',
}


def check(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            if not isinstance(json.loads(next(f)), dict):
                raise ValueError('Unexpected first record')
    else:
        data = json.loads(path.read_text())
        if data.get('type') != 'FeatureCollection' or not data.get('features'):
            raise ValueError('Expected nonempty GeoJSON feature collection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-directory', type=Path, help='Use existing downloads before network')
    args = parser.parse_args()
    target = ROOT / 'data/raw'
    target.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = target / name
        if path.exists():
            check(path)
            print(f'Existing input: {path}', flush=True)
            continue
        source = args.from_directory / name if args.from_directory else None
        if source and source.is_file():
            check(source)
            shutil.copy2(source, path)
            print(f'Imported: {source}', flush=True)
            continue
        partial = path.with_name(name + '.partial')
        print(f'Downloading {name} (full New York reviews are about 2.65 GB)', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open('wb') as out:
                expected = response.headers.get('Content-Length')
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            if expected is not None and partial.stat().st_size != int(expected):
                raise ValueError('Incomplete download; partial file retained')
            # Check under the final extension without renaming an invalid input.
            if name.endswith('.gz'):
                with gzip.open(partial, 'rt') as f:
                    json.loads(next(f))
            else:
                data = json.loads(partial.read_text())
                if not data.get('features'):
                    raise ValueError('Empty boundary response')
            partial.replace(path)
        except Exception as exc:
            raise SystemExit(f'Download failed for {name}: {exc}. '
                             'No training result produced. Supply a completed local download '
                             'or the genuine processed sample_interactions.parquet.') from exc


if __name__ == '__main__':
    main()
