"""把《预实验分析报告（平衡采样）》Markdown 渲染为单文件离线 HTML。

与 render_preexperiment_report_html.py（预实验结论深化分析）同一设计系统；
该脚本依赖 markdown + bs4，本脚本只依赖 markdown-it-py，且不处理图片
（本报告无图）。用法：python code/scripts/render_analysis_report_html.py
输出：outputs/analysis/预实验分析报告_平衡采样_2026-11-03.html
"""
import base64
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
STEM = "预实验分析报告_平衡采样_2026-11-03"
SOURCE = ROOT / "outputs" / "analysis" / f"{STEM}.md"
OUTPUT = SOURCE.with_suffix(".html")

SHORT_TITLES = ["为什么做这轮实验", "模型、数据与准备", "E1 看图选描述",
                "E2 图片变化怎样记录", "E3 哪组搭配更难",
                "三个实验说明什么", "下一步与详细数据"]

CSS = r"""
:root{--ink:#172b41;--muted:#5e7081;--line:#dce5ec;--paper:#fff;--back:#f3f6f9;--blue:#225b85;--accent:#c5793e;--nav:246px;--radius:14px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:34px}
body{margin:0;color:var(--ink);background:var(--back);font-family:Inter,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.9;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:thickness 1px;text-underline-offset:4px}a:hover{color:#a05b25}
button{font:inherit;cursor:pointer}:focus-visible{outline:3px solid #b77b40;outline-offset:4px}
.reading-progress{position:fixed;top:0;left:0;height:3px;background:#d49a5f;width:0;z-index:20;transition:width .1s linear}
.shell{max-width:1460px;margin:auto;padding:30px 38px 70px;display:grid;grid-template-columns:var(--nav) minmax(0,1fr);gap:38px}
.sidebar{align-self:start;position:sticky;top:30px;max-height:calc(100vh - 60px);overflow:auto;padding-right:12px;scrollbar-width:thin}
.brand{font-size:20px;font-weight:750;letter-spacing:-.7px;color:#193c59;text-decoration:none}.brand span{color:var(--accent)}
.sidebar .edition{font-size:12px;color:var(--muted);margin:4px 0 28px;letter-spacing:1px}
.toc-details>summary{list-style:none;font-size:12px;font-weight:700;letter-spacing:2px;color:#768795;margin-bottom:14px;cursor:default}.toc-details>summary::-webkit-details-marker{display:none}
.toc{display:grid;gap:4px}.toc a{display:grid;grid-template-columns:25px 1fr;gap:7px;text-decoration:none;color:#526576;font-size:13px;line-height:1.6;padding:9px 10px;border-radius:7px;border-left:2px solid transparent}
.toc a .num{font-variant-numeric:tabular-nums;font-size:11px;color:#8b99a6;line-height:1.9}.toc a:hover{background:#e8eff4;color:var(--ink)}
.toc a[aria-current="location"]{color:var(--blue);background:#e5eef5;border-left-color:var(--blue);font-weight:650}
.sidebar-footer{border-top:1px solid var(--line);margin-top:26px;padding:18px 10px 0;font-size:12px;color:var(--muted);line-height:1.9}.sidebar-footer a{font-size:12px}
main{min-width:0}.hero{border-radius:18px;background:#18354d;color:#fff;padding:42px 44px 36px;position:relative;overflow:hidden}
.hero:after{content:"Δ";font-family:Georgia,serif;font-size:230px;line-height:1;position:absolute;right:16px;top:40px;color:#ffffff08;pointer-events:none}
.eyebrow{font-size:11px;letter-spacing:2px;font-weight:600;color:#b2c4d2;margin:0 0 16px}
h1{font-size:clamp(28px,3.1vw,41px);font-weight:700;line-height:1.4;letter-spacing:-1px;margin:0 0 18px;max-width:770px;position:relative;z-index:1}
.hero-desc{font-size:15px;color:#c5d5e0;max-width:660px;margin:0;line-height:1.8}.hero-meta{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:26px;font-size:12px;color:#b3c7d6}.hero-meta span:before{content:"";display:inline-block;width:5px;height:5px;background:#d99b60;border-radius:50%;vertical-align:middle;margin-right:8px}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 26px}.metric{border:1px solid var(--line);background:#fff;border-radius:10px;padding:20px 22px}.metric .value{font-size:25px;letter-spacing:-.6px;font-weight:700;line-height:1.3;color:#244e70;font-variant-numeric:tabular-nums}.metric .unit{font-size:12px;margin-left:4px;font-weight:500;letter-spacing:0}.metric .label{font-size:12px;color:var(--muted);line-height:1.7;margin-top:9px}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 3px 18px}.toolbar-note{font-size:12px;color:#718393}.print-button{background:#fff;border:1px solid #cbd8e2;border-radius:7px;color:#264862;padding:7px 13px;font-size:12px;line-height:1.6}.print-button:hover{background:#eaf1f6}
.report{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);padding:10px 42px 28px;box-shadow:0 7px 28px #1c354606}
.report p{margin:16px 0;overflow-wrap:break-word}.report strong{font-weight:650;color:#173c59}.report li{margin:9px 0;padding-left:3px}.report ul,.report ol{padding-left:23px;margin:18px 0}
.intro{padding:10px 0 22px;border-bottom:1px solid var(--line);font-size:15px}
.report-section{padding:17px 0 26px;border-bottom:1px solid var(--line);scroll-margin-top:30px}.report-section:last-child{border-bottom:0;padding-bottom:5px}
h2{font-size:23px;line-height:1.55;letter-spacing:-.4px;margin:25px 0 20px;padding-left:16px;border-left:4px solid #5182a4;color:#173a56}h3{font-size:18px;line-height:1.65;margin:27px 0 13px;color:#304e65}
.table-scroll{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:9px;max-width:100%;scrollbar-width:thin}
table{border-collapse:collapse;min-width:100%;font-size:13px;line-height:1.85;font-variant-numeric:tabular-nums}thead{background:#edf3f7;color:#274b65}th{font-weight:650;text-align:left}th,td{padding:12px 15px;border-bottom:1px solid #e5ebf0;vertical-align:top}th:not(:first-child),td:not(:first-child){min-width:100px}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#f8fafc}tbody tr:hover{background:#eef4f8}td strong{white-space:nowrap}
code{background:#f0f3f6;border:1px solid #e4eaf0;border-radius:4px;padding:2px 5px;font-family:"Cascadia Code",Consolas,monospace;font-size:.83em;overflow-wrap:anywhere}
.report-footer{font-size:12px;color:#788895;margin:23px 4px 0;line-height:1.9;display:flex;justify-content:space-between;gap:20px}.report-footer a{color:#688094}
.report code{white-space:normal}.table-scroll table{width:100%}h4{font-size:16px;line-height:1.7;color:#304e65;margin:28px 0 12px}
@media(min-width:1600px){.shell{padding-top:42px}.sidebar{top:42px}}@media(max-width:1100px){:root{--nav:200px}.shell{gap:24px;padding:22px}.hero{padding:34px}.report{padding:8px 28px 24px}.metric{padding:17px}.metric .value{font-size:22px}th,td{padding:10px 12px}}
@media(max-width:820px){.shell{display:block;padding:18px 16px 36px}.sidebar{position:static;max-height:none;padding:0;margin-bottom:20px}.brand{font-size:19px}.sidebar .edition{display:inline-block;margin:0 0 0 15px;font-size:10px}.toc-details{margin-top:16px;background:#eaf0f5;border-radius:9px;padding:12px 14px}.toc-details>summary{margin:0;cursor:pointer;letter-spacing:1px}.toc-details>summary:after{content:"展开 / 收起";float:right;font-weight:400;font-size:10px;letter-spacing:0}.toc{margin-top:12px;grid-template-columns:1fr 1fr}.toc a{padding:7px 5px;font-size:12px}.sidebar-footer{display:none}.hero{padding:29px 25px}.hero h1{font-size:28px;letter-spacing:-.5px}.hero-desc{font-size:13px}.hero-meta{margin-top:21px;font-size:11px}.metrics{gap:9px;margin:13px 0 20px}.metric{padding:15px 12px}.metric .value{font-size:20px}.metric .label{font-size:10px}.report{padding:5px 22px 20px}.report-section{padding-top:12px}h2{font-size:21px}h3{font-size:17px}.report p,.report li{font-size:15px}.report-footer{display:block}}
@media(max-width:480px){.hero{padding:24px 20px}.hero h1{font-size:25px}.hero-meta{gap:5px 13px}.metrics{grid-template-columns:1fr}.metric{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:14px 17px}.metric .label{margin:0;font-size:11px;text-align:right;max-width:155px}.metric .value{font-size:21px;white-space:nowrap}.toolbar-note{font-size:10px}.report{padding:4px 16px 18px}.report-section{padding-bottom:17px}h2{font-size:19px;padding-left:10px}.table-scroll{margin:20px 0}.toc{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.reading-progress{transition:none}}
@page{size:A4;margin:18mm 15mm}
@media print{html{scroll-behavior:auto}body{background:#fff;font-size:10pt;line-height:1.65;color:#172b41}.shell{display:block;max-width:none;padding:0}.sidebar,.toolbar,.reading-progress{display:none!important}.hero{background:#fff!important;color:#173a56;padding:0 0 16pt;border-radius:0;border-bottom:2px solid #244e70;overflow:visible}.hero:after{display:none}.eyebrow{color:#5c7182;font-size:8pt;margin-bottom:8pt}h1{font-size:23pt;max-width:none}.hero-desc{color:#4e6273;font-size:10pt}.hero-meta{color:#617486;margin-top:12pt;font-size:8pt}.metrics{gap:10pt;margin:14pt 0}.metric{padding:10pt;border-color:#d6dfe7}.metric .value{font-size:17pt}.metric .label{font-size:8pt}.report{border:0;box-shadow:none;padding:0}.report p,.report li{font-size:10pt}.intro{font-size:10pt;padding-bottom:14pt}.report-section{padding:8pt 0 14pt;scroll-margin-top:0}h2{font-size:15pt;break-after:avoid;margin:16pt 0 12pt}h3{font-size:11.5pt;break-after:avoid;margin:16pt 0 8pt}.table-scroll{overflow:visible;border-radius:0;margin:14pt 0}table{font-size:8pt;min-width:100%;width:100%;table-layout:auto}th,td{padding:6pt 7pt;min-width:0!important;overflow-wrap:anywhere}thead{display:table-header-group}tr{break-inside:avoid}a{color:#245b80;text-decoration:none}a[href^="data:"]{text-decoration:underline}code{font-size:8pt}.report strong{color:inherit}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

JS = r"""
const tocDetails = document.querySelector('.toc-details');
if (matchMedia('(max-width:820px)').matches) tocDetails.open = false;
document.getElementById('print-report').addEventListener('click', () => window.print());
const sections = [...document.querySelectorAll('.report-section')];
const links = [...document.querySelectorAll('.toc a')];
let ticking = false;
function updateReading() {
  const extent = document.documentElement.scrollHeight - innerHeight;
  document.querySelector('.reading-progress').style.width = `${extent > 0 ? Math.min(100, scrollY/extent*100) : 100}%`;
  let active = '';
  for (const section of sections) if (section.getBoundingClientRect().top <= 140) active = section.id;
  for (const link of links) {
    if (link.hash === '#' + active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  ticking = false;
}
addEventListener('scroll', () => { if (!ticking) { requestAnimationFrame(updateReading); ticking = true; } }, {passive:true});
addEventListener('resize', updateReading);
updateReading();
"""


def render():
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# "), "报告第一行应为 H1"
    body_md = "\n".join(lines[1:]).strip()
    rendered = MarkdownIt("commonmark").enable("table").render(body_md)

    # 表格包滚动容器（生成的 HTML 无嵌套 table，替换安全）
    n_tables = 0

    def wrap_table(match):
        nonlocal n_tables
        n_tables += 1
        label = f"数据表 {n_tables}，可横向滚动"
        return (f'<div class="table-scroll" tabindex="0" role="region" '
                f'aria-label="{label}">{match.group(0)}</div>')

    rendered = re.sub(r"<table>.*?</table>", wrap_table, rendered, flags=re.S)

    # 按 h2 切 section；首段（无 h2 的开头 ul）作为 intro
    parts = re.split(r"(?=<h2>)", rendered)
    intro_html, section_htmls = parts[0], parts[1:]
    assert len(section_htmls) == len(SHORT_TITLES), \
        f"章节数 {len(section_htmls)} != 目录数 {len(SHORT_TITLES)}"
    sections = []
    for i, chunk in enumerate(section_htmls, 1):
        heading_id = f"section-{i}-heading"
        chunk = chunk.replace("<h2>", f'<h2 id="{heading_id}">', 1)
        sections.append(f'<section class="report-section" id="section-{i}" '
                        f'aria-labelledby="{heading_id}">{chunk}</section>')

    toc = "".join(
        f'<a href="#section-{i}"><span class="num">{i:02}</span>'
        f'<span>{html.escape(t)}</span></a>'
        for i, t in enumerate(SHORT_TITLES, 1))
    article = f'<div class="intro">{intro_html}</div>\n' + "\n".join(sections)
    source_download = ("data:text/markdown;base64,"
                       + base64.b64encode(SOURCE.read_bytes()).decode("ascii"))

    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="description" content="GraphDelta 预实验分析报告：说明为什么做实验、E1/E2/E3 每一步怎样做、发现什么问题，以及结论和下一步。">
<title>GraphDelta｜预实验分析报告（平衡采样）· 2026-11-03</title>
<style>{CSS}</style>
</head>
<body id="top">
<div class="reading-progress" aria-hidden="true"></div>
<div class="shell">
<aside class="sidebar" aria-label="报告目录">
  <a class="brand" href="#top">Graph<span>Delta</span></a>
  <div class="edition">预实验 · 实验过程与结果</div>
  <details class="toc-details" open>
    <summary>阅读目录</summary>
    <nav class="toc" aria-label="章节导航">{toc}</nav>
  </details>
  <div class="sidebar-footer">分析日期 · 2026.11.03<br>从研究动机讲到实验结论<br>依据已有结果重写说明<br><a href="{source_download}" download="{SOURCE.name}">下载 Markdown 分析稿</a></div>
</aside>
<main>
  <header class="hero">
    <p class="eyebrow">GRAPHDELTA · 第一阶段：先确认问题</p>
    <h1>预实验分析报告<br>实验怎么做，结果说明什么</h1>
    <p class="hero-desc">模型认出了物体和颜色，是否也分得清颜色属于谁、谁在谁的左边？<br>从研究动机出发，逐步说明 E1、E2、E3 的做法、发现和结论。</p>
    <div class="hero-meta"><span>2026 年 11 月 3 日</span><span>3 个模型 × 3 次抽样</span><span>平衡采样 · 模型参数保持不变</span></div>
  </header>
  <div class="metrics" aria-label="关键发现">
    <div class="metric"><div class="value">E1<span class="unit">看图选描述</span></div><div class="label">发现问题：颜色题较好<br>关系和颜色归属题明显更难</div></div>
    <div class="metric"><div class="value">E2<span class="unit">比较图片变化</span></div><div class="label">检查内部变化：环境最清楚<br>尚未直接检查关系与归属变化</div></div>
    <div class="metric"><div class="value">E3<span class="unit">比较搭配难度</span></div><div class="label">检查组合差异：两个模型在 B 更难<br>还未验证没见过的组合能否答对</div></div>
  </div>
  <div class="toolbar"><span class="toolbar-note">动机 → 实验步骤 → 发现的问题 → 结论 · 详细数值见文末</span><button class="print-button" id="print-report" type="button">打印 / 保存 PDF</button></div>
  <article class="report" aria-label="分析报告正文">{article}</article>
  <footer class="report-footer"><span>GraphDelta · 预实验分析报告（平衡采样）· 2026-11-03<br>依据已有实验结果重写说明；本次未重新运行模型。数据来源与限制见第 6 节。</span><a href="#top">返回顶部 ↑</a></footer>
</main>
</div>
<script>{JS}</script>
</body>
</html>'''
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"HTML: {OUTPUT}")
    print(f"Sections: {len(sections)}; tables: {n_tables}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    render()
