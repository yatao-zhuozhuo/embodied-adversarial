#!/usr/bin/env python3
"""Prepare short expert episodes plus successful online Gemini/GPT exports."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'SFT_data'
OUT = ROOT / 'mixed-data'

def main():
    paths = sorted((SOURCE / 'expert_replay_compressed_reasoning_v1/exports').glob('*/train.v1.jsonl'))
    paths += sorted(p for p in SOURCE.glob('*/exports/*/train.v1.jsonl')
                    if p.parts[-4].startswith(('gemini_', 'codingrouter_', 'gpt_')))
    rows, summary, seen = [], [], set()
    for path in paths:
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                groups[row['episode_id']].append(row)
        expert = path.parts[-4] == 'expert_replay_compressed_reasoning_v1'
        kept, excluded, count = 0, 0, 0
        for episode, group in groups.items():
            # Previously rejected one-turn Flash StackCans trajectory.
            if not expert and path.parts[-4] == 'gemini_flash_expanded_20261006_single100' and path.parent.name == 'robocasa':
                excluded += 1
                continue
            turns = max(len(group), max(row['metadata']['source_turn'] for row in group))
            if expert and turns > 100:
                excluded += 1
                continue
            kept += 1
            for row in group:
                key = (episode, row['metadata']['source_turn'])
                if key in seen:
                    continue
                seen.add(key)
                assert row['metadata']['official_episode_success'] is True
                messages = row['messages']
                assert messages[-1]['role'] == 'assistant' and messages[-1]['loss'] is True
                assert all(m.get('loss') is False for m in messages[:-1] if m['role'] == 'assistant')
                assert sum(m['content'].count('<image>') for m in messages) == len(row['images'])
                assert all(Path(p).is_file() for p in row['images'])
                rows.append(row)
                count += 1
        summary.append({'source': str(path), 'episodes': kept, 'excluded_long_episodes': excluded, 'samples': count})
    OUT.mkdir(exist_ok=True)
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    result = {'samples': len(rows), 'episodes': len({r['episode_id'] for r in rows}),
              'samples_by_bench': dict(Counter(r['metadata']['bench'] for r in rows)), 'sources': summary}
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
