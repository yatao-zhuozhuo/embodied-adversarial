#!/usr/bin/env python3
"""Copy the five successful LIBERO/MetaWorld episodes' exported decisions."""
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'SFT_data/gemini_flash_expanded_20261006_single100'


def main():
    rows = []
    for bench in ('libero', 'metaworld'):
        source = SOURCE / 'exports' / bench / 'train.v1.jsonl'
        rows.extend(json.loads(line) for line in source.read_text().splitlines() if line.strip())
    assert len(rows) == 133
    assert len({row['episode_id'] for row in rows}) == 5
    copied = {}
    for row in rows:
        assert row['metadata']['official_episode_success'] is True
        assert row['metadata']['bench'] in ('libero', 'metaworld')
        assert row['messages'][-1]['role'] == 'assistant'
        assert row['messages'][-1]['loss'] is True
        assert all(m.get('loss') is False for m in row['messages'][:-1] if m['role'] == 'assistant')
        assert sum(m['content'].count('<image>') for m in row['messages']) == len(row['images'])
        images = []
        for image in row['images']:
            original = Path(image).resolve()
            destination = ROOT / 'data/images' / original.relative_to(SOURCE.resolve())
            if original not in copied:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                copied[original] = destination
            images.append(str(destination))
        row['images'] = images
    output = ROOT / 'data/train.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'episodes': 5, 'samples': len(rows), 'images': len(copied),
               'samples_by_bench': dict(Counter(row['metadata']['bench'] for row in rows)),
               'source_episodes': sorted({row['episode_id'] for row in rows})}
    (ROOT / 'data/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
