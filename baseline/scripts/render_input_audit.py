"""Export one canonical input snapshot as a standalone, model-visible-only HTML."""
import argparse
import base64
import html
import json
from pathlib import Path


def render(package):
    panels=[]
    for e in package['images']:
        data=base64.b64encode(Path(e['path']).read_bytes()).decode()
        label=html.escape(json.dumps({k:v for k,v in e.items() if k!='path'},ensure_ascii=False,indent=2))
        panels.append(f'<figure><img src="data:image/png;base64,{data}"><figcaption><pre>{label}</pre></figcaption></figure>')
    visible={k:v for k,v in package.items() if k!='images'}
    return '<!doctype html><meta charset="utf-8"><title>本轮模型实际输入</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}figure{display:inline-block;vertical-align:top;max-width:400px}</style><h1>本轮模型实际输入</h1><p>图片按实际发送顺序排列。仅显示模型输入，不含宿主评分或完整环境状态。</p>'+''.join(panels)+'<h2>文本、反馈与历史</h2><pre>'+html.escape(json.dumps(visible,ensure_ascii=False,indent=2))+'</pre>'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(render(json.loads(a.snapshot.read_text())))
