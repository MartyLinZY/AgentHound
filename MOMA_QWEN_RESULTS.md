# MoMA Qwen3.5-27B baseline 实测（2026-10-05）

已完成8条短轨迹All-at-Once试跑，根因步骤精确命中1/8（12.5%），模型响应均为HTTP 200、finish_reason=stop。此结果不是RootSE完整102条成绩。

## 设置与范围

- 用户指定移动云MoMA；端点：https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions。
- 同源models接口确认请求模型ID `Qwen3.5-27B`，实际响应名 `qwen3d5-27b-32k`。
- 官方TrajAudit All-at-Once提示词；任务、完整原始轨迹、首条失败测试code/error。不提供gold patch或根因标签。
- temperature=0，max_tokens=8192，非流式普通文本模式。数据和提示词哈希保存在manifest/records。
- 按实际模型输入字符数选择最短8条，无标签参与选择，均来自AutoCodeRover/gpt-4-0125。未截断轨迹，因此存在明确的短轨迹和agent分布偏差。
- 7条为合法JSON，1条解释文字包含未转义引号；仅在严格两字段外壳和独立合法步骤ID成立时恢复解释文字，原始输出保留。恢复样本未命中，不改变本次命中数。

## 结果

| 样本 | 预测步骤 | 标注根因步骤 | 命中 | 格式恢复 |
|---|---:|---|---|---|
| instance_astropy__astropy-14182 | 2 | 1 | 否 | 是 |
| instance_django__django-14534 | 2 | 1 | 否 | 否 |
| instance_django__django-14787 | 2 | 1 | 否 | 否 |
| instance_django__django-13551 | 2 | 2 | 是 | 否 |
| instance_django__django-15252 | 2 | 1 | 否 | 否 |
| instance_django__django-15695 | 2 | 1 | 否 | 否 |
| instance_sympy__sympy-12419 | 3 | 1 | 否 | 否 |
| instance_django__django-14999 | 3 | 2 | 否 | 否 |

本次8条用量：50,839 tokens；逐请求耗时合计106.34秒。原始响应、逐条预测和metrics位于 `outputs/moma-qwen35-27b-short8-run1/`。

## 调试与服务限制

1. `pilot-run1`、`pilot-run2`各发送首条样本一次，HTTP 200，但JSON含未转义引号。分别报告4,783和4,891 tokens；run2保存了原始响应。没有为这些未完成运行生成准确率。
2. `pilot-run3`使用JSON object输出约束，首条请求180秒ReadTimeout，未取得usage。不能推断服务端是否完成生成或是否计费。
3. `context-probe`对原定分组pilot的第二条完整长轨迹请求返回HTTP 400，服务明确说明最大上下文32768 tokens，并报告输入至少24577 tokens加8192输出预算已超限。该下界不是输入的完整token数。进一步将输出预算降到1 token，服务仍返回400并报告输入至少32768 tokens；因此单纯降低输出预算不能解决这条完整轨迹的输入超限。
4. 共发送13次chat请求（8次最终试跑、3次格式调试、2次上下文诊断），收到usage的请求合计60,513 tokens；另有超时请求用量未知，以上不是完整账单。models查询不计入chat请求数。

目前无法按不截断的All-at-Once协议在此部署上覆盖RootSE全量。后续全量对照需要长上下文部署，或单独定义分块/逐步诊断方法；后者必须单列baseline，并与AgentHound使用一致模型与事后测试信息。不能直接把12.5%与此前完整102条上的Claude历史结果比较。

## 复现与验证

见 `QWEN_BASELINE.md`。入口支持本地密钥文件，密钥值不写入命令、manifest或输出。已检查本轮产物未包含密钥。解析/隔离/请求计数等56项测试通过。未调用Claude，也未向远端推送代码。
