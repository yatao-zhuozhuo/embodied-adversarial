"""Submit two 8-GPU RJobs sharing a frozen curriculum plan on GPFS."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def request(name, rank, root, acceptance, excluded_nodes=()):
    namespace, quota = 'fdj-infra', 'fdj_gpu_task'
    labels = {'brainpp-log-collector':'fluent-bit', 'kubebrain.brainpp.cn/creator':'yangyongzhuo',
        'kubebrain.brainpp.cn/creator-id':'e344b3ff5628d23c1f4ad3d234d89309',
        'kubebrain.brainpp.cn/creator-type':'user','kubebrain.brainpp.cn/creator-department':'default',
        'kubebrain.brainpp.cn/resourcetype':'rjob','privatemachine.brainpp.cn/privatemachine':'group',
        'quotagroup.brainpp.cn/preemptible':'no','quotagroup.brainpp.cn/quotagroup':quota,
        'rjob.brainpp.cn/scheduler-name':'volcano','rjob.brainpp.cn/zs-task-type':'rjob-normal',
        'scheduling.volcano.sh/queue-name':'fdj-infra-fdj-gpu-task'}
    annotations = {'kubebrain.brainpp.cn/module':quota,'kubebrain.brainpp.cn/showname':name,
        'qos.quality':'Guaranteed','rjob.brainpp.cn/scheduler-name':'volcano',
        'rjob.brainpp.cn/zs-task-type':'rjob-normal','scheduling.volcano.sh/queue-name':'fdj-infra-fdj-gpu-task',
        'volcano.brainpp.cn/priority':'5'}
    resources = {'cpu':'64','memory':'768Gi','nvidia.com/gpu':'8'}
    env = {'RUN_ROOT':str(root),'NODE_RANK':str(rank),'NNODES':'2','MASTER_PORT':'29885',
           'OPENETA_RL_MCP_PORT':'19785','ORIGINAL_ACCEPTANCE':str(acceptance),
           'NCCL_DEBUG':'WARN','PYTHONUNBUFFERED':'1'}
    command = 'cd '+shlex.quote(str(ROOT))+' && bash scripts/run_single_model_rl_yi_job.sh'
    body = {'apiVersion':'rjob.brainpp.cn/v1alpha1','kind':'RJob',
        'metadata':{'name':name,'namespace':namespace,'labels':labels,'annotations':annotations},
        'spec':{'autoDeleteDuration':'720h','backoffLimit':1,'failurePolicy':'None','taskSpecs':{'task1':{
            'maxRunningDuration':'72h','maxWaitDuration':'24h','positiveTags':['feature/gpfs=yes'],
            'replicas':1,'rootfsSource':{'kind':'image','metadata':{'creationTimestamp':None}},'updatePolicy':'Active',
            'template':{'metadata':{'labels':labels,'annotations':{'kubebrain.brainpp.cn/module':quota,
                'mount.brainpp.cn/gpfs':'gpfs://gpfs1/self-evolve:/mnt/shared-storage-user/self-evolve'}},
                'spec':{'hostNetwork':True,'dnsPolicy':'ClusterFirstWithHostNet','containers':[{
                    'name':'task1','image':'registry.lg.shzhisuan.com/library/sglang:v0.5.12.post1-cu130',
                    'command':['bash','-lc',command],'env':[{'name':k,'value':v} for k,v in env.items()],
                    'resources':{'limits':resources,'requests':resources}}],
                    'imagePullSecrets':[{'name':labels['kubebrain.brainpp.cn/creator-id']}],
                    'restartPolicy':'Never','terminationGracePeriodSeconds':120}}}}}}
    if excluded_nodes:
        body['spec']['taskSpecs']['task1']['template']['spec']['affinity'] = {
            'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {'nodeSelectorTerms': [
                {'matchExpressions': [{'key': 'kubernetes.io/hostname', 'operator': 'NotIn',
                                       'values': list(excluded_nodes)}]}]}}}
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--original-acceptance',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--exclude-node',action='append',default=[])
    args=parser.parse_args()
    root=args.run_root.resolve()
    if not (root/'curriculum_plan.json').is_file():parser.error('Prepare the curriculum first')
    directory=root/'jobs';directory.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
    for rank in range(2):
        name=f'{args.name}-{rank}'
        path=directory/f'{name}.json'
        body=request(name,rank,root,args.original_acceptance.resolve(),args.exclude_node)
        path.write_text(json.dumps(body,indent=2)+'\n')
        if args.prepare_only:
            print(path);continue
        receipt=directory/f'{name}.receipt.json'
        if receipt.exists():raise RuntimeError('Existing submission receipt; inspect before resubmission')
        subprocess.run(['brainctl','create','-f',str(path),'--request-timeout=30s'],env=env,check=True)
        live=json.loads(subprocess.check_output(['brainctl','get','rjob',name,'-n','fdj-infra','-o','json','--request-timeout=30s'],env=env))
        record={'name':name,'uid':live['metadata']['uid'],'created_at':live['metadata']['creationTimestamp'],
                'status':live.get('status',{}),'node_rank':rank}
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
