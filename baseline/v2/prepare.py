"""Build the v2 planning artifacts; writes only beside this script, never runs evaluations."""
import argparse
import collections
import datetime
import hashlib
import json
import re
import runpy
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = Path(__file__).resolve().parent
    baseline = out.parent
    train_path = root / 'SFT-train/mixed-data/train.jsonl'
    legacy_path = root / 'docs/baseline-40-tasks.json'
    mapping_path = baseline / '.runtime/sources/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'
    mapping = runpy.run_path(str(mapping_path))['libero_task_map']
    rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    legacy = json.loads(legacy_path.read_text())['episodes']

    def canonical(env_id):
        match = re.fullmatch(r'openeta/libero_(libero_\w+)_task(\d+)-v0', env_id)
        if match:
            return 'libero:' + mapping[match[1]][int(match[2])]
        return re.sub(r'^openeta/', '', env_id)

    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['metadata']['env_id']].append(row)
    tasks = []
    for env_id, samples in sorted(grouped.items()):
        tasks.append({
            'env_id': env_id,
            'benchmark': samples[0]['metadata']['bench'],
            'canonical_key': canonical(env_id),
            'episode_count': len({s['episode_id'] for s in samples}),
            'decision_sample_count': len(samples),
            'seeds': sorted({s['metadata']['seed'] for s in samples}),
            'source_episodes': sorted({s['episode_id'] for s in samples}),
            'exclusion_scope': 'all_initial_states_of_this_task',
        })
    provenance = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'training_file': str(train_path), 'training_sha256': digest(train_path),
        'legacy_file': str(legacy_path), 'legacy_sha256': digest(legacy_path),
        'libero_mapping_file': str(mapping_path), 'libero_mapping_sha256': digest(mapping_path),
    }
    exclusions = {
        'schema_version': 'openeta.baseline.v2.training_exclusions.v1',
        'provenance': provenance,
        'scope': '当前 mixed-data/train.jsonl；正式冻结前须合并所有参评 checkpoint 的训练任务清单',
        'episode_count': len({s['episode_id'] for s in rows}),
        'decision_sample_count': len(rows), 'task_count': len(tasks), 'tasks': tasks,
    }
    train_keys = {t['canonical_key'] for t in tasks}
    exact = [e['episode_id'] for e in legacy if e['env_id'] in grouped]
    canonical_overlap = [e['episode_id'] for e in legacy if canonical(e['env_id']) in train_keys]
    audit = {
        'schema_version': 'openeta.baseline.v2.overlap_audit.v1',
        'provenance': provenance, 'evaluated_task_count': len(legacy),
        'exact_env_id_overlap': exact, 'libero_name_normalized_overlap': canonical_overlap,
        'status': 'preliminary_not_semantic_clearance',
        'pending_checks': [
            'LIBERO BDDL 场景、目标谓词和对象角色的语义近重复检查',
            'RoboCasa 不同注册前缀对应的原生任务类归一化',
            '所有参评 checkpoint 的训练来源并集',
            '固定初始化索引、轨迹和图片指纹检查；不同 seed 不保证不同初始状态',
        ],
    }
    plan = {
        'schema_version': 'openeta.baseline.v2.expansion_plan.v1',
        'status': 'planning_not_frozen', 'language': 'zh-CN',
        'purpose': '扩展任务覆盖和难度梯度，评估未训练任务上的泛化能力',
        'storage_rule': 'v2 题单、配置、脚本、审计、快照、日志和结果均保存在本目录下；已有数据仅引用，不修改',
        'target': {
            'task_count': 72, 'distinct_initial_states_per_task': 3, 'episode_count': 216,
            'tasks_by_benchmark': {'libero': 20, 'metaworld': 20, 'maniskill': 12, 'robocasa': 20},
            'difficulty_proportions': {'easy': 0.4, 'medium': 0.4, 'hard': 0.2},
            'note': '规划目标，最终按排除训练任务和环境验收后的可用数量调整；不通过换机器人或重复 seed 凑任务数',
            'later_distinct_initial_states_per_task': 5,
        },
        'difficulty': {
            'label_source': 'OpenETA 自定义，非官方统一分级',
            'easy': '短流程、目标清楚、定位或接触容差较大',
            'medium': '抓取搬运、释放放置、空间关系辨认、少量阶段组合',
            'hard': '精密插入、工具使用、复杂堆叠、多物体或长流程',
            'calibration': '按阶段数、精度、遮挡和导航要求预分类，再用相同控制接口下的参考执行校准；不按被测模型输赢定难度',
        },
        'split_rules': [
            '测试任务与所有参评模型训练任务并集互斥；同一任务换 seed 仍排除',
            '规范化原生任务定义，不能仅比较 env_id 或文字描述',
            '额外标注与训练集相近的任务家族；共享基本技能不自动视为数据泄漏',
            '调试开发题与冻结正式测试题分开；正式测试题不再用于训练或调参',
        ],
        'acceptance_gates': [
            '训练任务排除及语义近重复审计通过',
            '环境版本、机器人、相机和动作接口固定',
            '目标可从授权输入图像或任务文字获知',
            '原生成功判定通过正反例检查',
            '参考执行验证接口和预算可行；特权状态参考策略不代表视觉模型难度',
            '同时固定模型决策轮数和模拟器控制步数，核对原生 horizon；同题各模型预算相同',
            '固定真实初始状态而不只是 seed，冻结输入输出协议和文件哈希',
        ],
        'reporting': [
            '原生成功率按 benchmark、难度和任务分别统计',
            '总分先按任务平均再按 benchmark 等权平均，避免配额不同造成偏重',
            '统计格式错误、动作拒绝、基础设施错误、决策预算耗尽和环境 horizon',
            '有可靠判定时补充阶段进展，不替代原生成功率',
            '记录 checkpoint 与其训练数据指纹，区分五轨迹旧模型和 74 轨迹混合训练模型',
        ],
        'implementation_order': [
            '冻结训练排除范围', '筛选候选任务并标注初步难度',
            '完成环境及预算验收', '分配独立初始状态并冻结正式题单',
            '用相同协议评测基座模型和 SFT 模型',
        ],
        'official_references': {
            'libero': 'https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/code_structure.html',
            'metaworld': 'https://metaworld.farama.org/benchmark/benchmark_descriptions/',
            'maniskill': 'https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html',
            'robocasa': 'https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html',
        },
    }
    candidates = {
        'schema_version': 'openeta.baseline.v2.candidate_pool.v1',
        'status': 'legacy_candidates_only_not_runnable_manifest',
        'note': '现有 40 题仅作为候选起点；新增题目尚未筛选，难度和语义排除尚未验收',
        'provenance': provenance,
        'tasks': [{
            'candidate_id': e['episode_id'], 'env_id': e['env_id'], 'task': e['task'],
            'benchmark': e['metadata']['benchmark'], 'canonical_key': canonical(e['env_id']),
            'source': 'legacy_baseline_40', 'difficulty': None,
            'acceptance_status': 'pending', 'evaluation_initial_states': [],
        } for e in legacy],
    }
    for filename, obj in [('plan.json', plan), ('training_exclusions.json', exclusions),
                          ('overlap_audit.json', audit), ('candidate_pool.json', candidates)]:
        (out / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'training_tasks': len(tasks),
                      'training_episodes': exclusions['episode_count'],
                      'training_samples': len(rows), 'legacy_candidates': len(legacy),
                      'exact_overlap': exact, 'normalized_overlap': canonical_overlap}, ensure_ascii=False))


if __name__ == '__main__':
    main()
