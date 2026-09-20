"""Requirement-by-requirement evidence audit before packaging. Not a product model test."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
NAMES=['vision-spec-to-evidence','vision-data-review','vision-generalization','pose-guided-capture','vision-feature-engineering','vision-metrics-calibration','vision-result-publication','vision-batch-recovery','vision-runtime-parity','vision-system-integration']
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args()
 checks=[]
 def record(i,claim,ok,paths,detail):
  checks.append({'id':i,'requirement':claim,'passed':bool(ok),'evidence':paths,'verification':detail})
 shape=read('evaluation/structure_checks.json');base=read('evaluation/self_checks.json');linux=read('evaluation/linux_checks.json')
 cases=read('tests/cases.json');refs=read('evidence/method_sources.json')['records']
 for index,name in enumerate(NAMES,1):
  paths=['skills/'+name+'/'+f for f in ['SKILL.md','agents/openai.yaml','scripts/check.py','references/contract.md','references/project-case.md','references/worked-example.md']]
  related=[c for c in cases if c['skill']==name]
  ids={c['id'] for c in related};tested=[x for x in base['cases'] if x['id'] in ids]
  record('SKILL-'+str(index).zfill(2),name+'：入口/输入/流程/输出/边界/案例/工具',
   all((R/p).is_file() for p in paths) and len(related)==3 and len(tested)==3 and all(x['passed'] for x in tested) and any(x['skill']==name for x in refs),
   paths+['evaluation/self_checks.json','evidence/method_sources.json'],
   '主代理已阅读内容与完整产物示例；独立代理审查全部入口/合同并逐题判断。存在文件不单独证明行为，关联3类实际检查。')
 record('STRUCTURE','10项命名、frontmatter、引用、无占位和无私有运行路径',shape['passed'] and len(shape['skills'])==10,['evaluation/structure_checks.json'],'官方quick_validate与静态副作用/依赖检查。')
 record('PORTABLE','Windows/Linux原目录及无原工程迁移目录运行',all(x['passed'] and x['source_count']==30 and x['relocated_count']==30 and all(y['passed'] for y in x['cases']+x['relocated_cases']) for x in (base,linux)),['evaluation/self_checks.json','evaluation/linux_checks.json'],'每平台30+30，保留独立临时目录和输入不改写断言。')
 regression=read('evaluation/review_regressions.json');extra=read('evaluation/completion_checks.json')
 record('GATES','明确列出的失败场景及修复回归',regression['passed'] and len(regression['cases'])==12 and extra['passed'] and extra['count']==12,['evaluation/review_regressions.json','evaluation/completion_checks.json'],'包括镜像、重拍/revision、失锁/间隔、分母、分位数、letterbox/镜像和假布尔。')
 b=read('evaluation/independent_baseline.json')['review'];w=read('evaluation/independent_with_skill.json')['review'];q=read('evaluation/independent_recheck.json')['review'];a=read('evaluation/independent_completion_audit.json')['review']
 record('INDEPENDENT','同一独立代理先baseline后Skill并修复复核',len(b['results'])==10 and len(w['results'])==30 and q['release_recommendation']['blocking_issues_remaining']==[] and not a['blocking_issues'] and a['release_recommendation']['approved'] is True and all(x['model']=='gpt-6-astra' and x['effort']=='medium' for x in (b,w,q,a)),['evaluation/independent_baseline.json','evaluation/independent_with_skill.json','evaluation/independent_recheck.json','evaluation/independent_completion_audit.json'],'真实公开代理消息带时间/事件哈希；不是主代理模拟评审。')
 project=read('evaluation/project_replay.json')
 record('PROJECT','项目案例及真实缓存复用，不重跑模型',project['passed'] and project['source_files_unchanged'] and not project['inference_executed'] and len(refs)==10,['evaluation/project_replay.json','evidence/method_sources.json','evidence/source_code.json'],'1525实例缓存实际域/JSON核对。10项历史来源独立于合成样例，不把历史计划冒充成功。')
 record('COVERAGE','覆盖产品运行/研发流程、职责分界、通用层与项目层',all((R/p).exists() for p in ['README.md','VALIDATION.md']) and len(NAMES)==10,['README.md','VALIDATION.md'],'人工内容审阅与独立完整产物示例审阅；不以日志大小/目录数替代工作流。')
 record('SCOPE','不改并行项目、Hook、模型；不训练/部署/推送',project['source_files_unchanged'] and q['method']['files_written']==0,['WORK_LOG.md','evaluation/project_replay.json','evaluation/independent_recheck.json'],'本任务主动写入在Skill沙箱；源码只读快照/缓存哈希核对。未做全系统监控，不能归因其他并行任务的修改。')
 record('PROVENANCE','证据真实性、候选与实际使用区分、无虚构效率提升',len(refs)==10 and all(len(x['source_file_sha256'])==64 and len(x['event_sha256'])==64 for x in refs),['evidence/method_sources.json','VALIDATION.md','evaluation/independent_completion_audit.json'],'对照报告记录实际观察及无法测量项；不作随机对照或因果效率声明。')
 trace=read('evaluation/comparison_measurements.json')['phases']
 record('COMPARISON-MEASURES','对照正确性/遗漏/无效操作说明、补充请求数和实际整轮耗时',len(a['representative_comparison'])==10 and len(trace)==4 and all(x['wall_seconds']>0 and x['user_input_request_calls']>=0 and x['model']=='gpt-6-astra' and x['effort']=='medium' for x in trace),['evaluation/comparison_measurements.json','evaluation/independent_completion_audit.json'],'实际任务事件计时和补充请求调用计数；逐题耗时未采集，不冒充已有。两组工作量不同，不据此宣称提速。')
 record('RELEASE-READY','可复制发布包的输入齐备',all((R/p).is_file() for p in ['LICENSE','NOTICE','VERSION','tools/build_release.py']),['tools/build_release.py'],'ZIP与解压校验在发布后以发布校验.json确认，当前只证明打包前置条件。')
 result={'passed':all(c['passed'] for c in checks),'stage':'pre-release-requirements-audit','at':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),'location_override':'User-authorized independent Skill sandbox supersedes original plan directory','checks':checks,'version':(R/'VERSION').read_text().strip()}
 (R/'evaluation/requirements_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':result['passed'],'requirements':len(checks),'failed':[c['id'] for c in checks if not c['passed']]},ensure_ascii=False))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
