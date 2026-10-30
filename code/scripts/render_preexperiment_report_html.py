"""Export the reviewed analysis as one offline HTML document.

Uses the Markdown report as the content source. Images and referenced supporting
files are embedded so the output remains usable when copied on its own.
"""
import base64
import html
import mimetypes
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
STEM = "预实验结论深化分析_证据强度与机制边界_2026-10-28"
SOURCE = ROOT / "outputs" / "analysis" / f"{STEM}.md"
OUTPUT = SOURCE.with_suffix(".html")

CSS = r"""
:root{--ink:#172b41;--muted:#5e7081;--line:#dce5ec;--paper:#fff;--back:#f3f6f9;--blue:#225b85;--accent:#c5793e;--nav:246px;--radius:14px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:34px}
body{margin:0;color:var(--ink);background:var(--back);font-family:Inter,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.9;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:4px}a:hover{color:#a05b25}
button,summary{font:inherit}button{cursor:pointer}:focus-visible{outline:3px solid #b77b40;outline-offset:4px}
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
.intro{padding:10px 0 22px;border-bottom:1px solid var(--line);font-size:15px}.intro>p:first-child{color:#5a6e7f}
.key-finding{background:#eef4f8;border-left:3px solid #42799f;border-radius:0 9px 9px 0;padding:19px 22px;line-height:1.95;margin:24px 0!important;color:#23455f}
.key-finding strong{font-weight:550}.report-section{padding:17px 0 26px;border-bottom:1px solid var(--line);scroll-margin-top:30px}.report-section:last-child{border-bottom:0;padding-bottom:5px}
h2{font-size:23px;line-height:1.55;letter-spacing:-.4px;margin:25px 0 20px;padding-left:16px;border-left:4px solid #5182a4;color:#173a56;text-wrap:pretty}h3{font-size:18px;line-height:1.65;margin:27px 0 13px;color:#304e65}
.table-scroll{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:9px;max-width:100%;scrollbar-width:thin}
table{border-collapse:collapse;min-width:100%;font-size:13px;line-height:1.85;font-variant-numeric:tabular-nums}thead{background:#edf3f7;color:#274b65}th{font-weight:650;text-align:left}th,td{padding:12px 15px;border-bottom:1px solid #e5ebf0;vertical-align:top}th:not(:first-child),td:not(:first-child){min-width:100px}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#f8fafc}tbody tr:hover{background:#eef4f8}td strong{white-space:nowrap}.structural-row{background:#f0f5f8!important}.evidence-table td:first-child{min-width:205px}.evidence-table td:last-child{min-width:225px}
.status{font-size:11px;font-weight:600;border-radius:4px;padding:3px 7px;display:inline-block;white-space:nowrap}.status.strong{background:#e4f1e9;color:#286249}.status.pending{background:#edf1f4;color:#5c6e7d}.status.limited{background:#fbf0df;color:#8a5d20}
code{background:#f0f3f6;border:1px solid #e4eaf0;border-radius:4px;padding:2px 5px;font-family:"Cascadia Code",Consolas,monospace;font-size:.83em;overflow-wrap:anywhere}pre{padding:18px;background:#f0f3f6;border-radius:8px;overflow:auto}pre code{padding:0;border:0}
figure{margin:26px 0;border:1px solid var(--line);border-radius:10px;padding:12px;background:#fff}figure img{display:block;width:100%;height:auto;border-radius:5px}.figure-tools{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:10px}.figure-tools span{font-size:11px;color:#708291;letter-spacing:1px}.zoom-button{background:transparent;border:0;padding:3px 7px;font-size:12px;color:var(--blue)}figcaption{font-size:12px;line-height:1.85;color:var(--muted);border-top:1px solid #e8edf2;padding-top:11px;margin-top:8px}
.equation{padding:22px 24px;margin:23px 0;background:#f5f8fa;border:1px solid #e0e8ee;border-radius:8px;overflow-x:auto;color:#244b68;font-family:"Cambria Math",Cambria,"Times New Roman",serif;font-size:20px;line-height:2.1}.equation-line{display:block;white-space:nowrap}.equation var{font-style:italic}.equation sub{font-size:65%}.equation .indent{padding-left:28px}
.source-section{font-size:13px;line-height:1.9}.source-section li{overflow-wrap:anywhere}.source-section code{font-size:11px}.report-footer{font-size:12px;color:#788895;margin:23px 4px 0;line-height:1.9;display:flex;justify-content:space-between;gap:20px}.report-footer a{color:#688094}
dialog{border:0;border-radius:12px;padding:18px;max-width:96vw;width:1450px;box-shadow:0 25px 100px #10283955}dialog::backdrop{background:#132d48ba}dialog img{width:100%;height:auto}dialog .dialog-top{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:13px;color:#526b7c;margin-bottom:16px}dialog .close-button{border:1px solid #dce5ec;background:#fff;border-radius:6px;padding:4px 12px;color:#324e63;font-size:12px}
@media(min-width:1600px){.shell{padding-top:42px}.sidebar{top:42px}}@media(max-width:1100px){:root{--nav:200px}.shell{gap:24px;padding:22px}.hero{padding:34px}.report{padding:8px 28px 24px}.metric{padding:17px}.metric .value{font-size:22px}th,td{padding:10px 12px}}
@media(max-width:820px){.shell{display:block;padding:18px 16px 36px}.sidebar{position:static;max-height:none;padding:0;margin-bottom:20px}.brand{font-size:19px}.sidebar .edition{display:inline-block;margin:0 0 0 15px;font-size:10px}.toc-details{margin-top:16px;background:#eaf0f5;border-radius:9px;padding:12px 14px}.toc-details>summary{margin:0;cursor:pointer;letter-spacing:1px}.toc-details>summary:after{content:"展开 / 收起";float:right;font-weight:400;font-size:10px;letter-spacing:0}.toc{margin-top:12px;grid-template-columns:1fr 1fr}.toc a{padding:7px 5px;font-size:12px}.sidebar-footer{display:none}.hero{padding:29px 25px}.hero h1{font-size:28px;letter-spacing:-.5px}.hero-desc{font-size:13px}.hero-meta{margin-top:21px;font-size:11px}.metrics{gap:9px;margin:13px 0 20px}.metric{padding:15px 12px}.metric .value{font-size:20px}.metric .label{font-size:10px}.report{padding:5px 22px 20px}.report-section{padding-top:12px}h2{font-size:21px}h3{font-size:17px}.report p,.report li{font-size:15px}.intro>p{font-size:14px}.key-finding{padding:14px 16px}.report-footer{display:block}.source-section p,.source-section li{font-size:12px}}
@media(max-width:480px){.hero{padding:24px 20px}.hero h1{font-size:25px}.hero-meta{gap:5px 13px}.metrics{grid-template-columns:1fr}.metric{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:14px 17px}.metric .label{margin:0;font-size:11px;text-align:right;max-width:155px}.metric .value{font-size:21px;white-space:nowrap}.toolbar-note{font-size:10px}.report{padding:4px 16px 18px}.report-section{padding-bottom:17px}h2{font-size:19px;padding-left:10px}.equation{font-size:17px;padding:16px}.toc{grid-template-columns:1fr}.table-scroll{margin:20px 0}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.reading-progress{transition:none}}
@page{size:A4;margin:18mm 15mm}
@media print{html{scroll-behavior:auto}body{background:#fff;font-size:10pt;line-height:1.65;color:#172b41}.shell{display:block;max-width:none;padding:0}.sidebar,.toolbar,.reading-progress,.zoom-button,.figure-tools,dialog{display:none!important}.hero{background:#fff!important;color:#173a56;padding:0 0 16pt;border-radius:0;border-bottom:2px solid #244e70;overflow:visible}.hero:after{display:none}.eyebrow{color:#5c7182;font-size:8pt;margin-bottom:8pt}h1{font-size:23pt;max-width:none}.hero-desc{color:#4e6273;font-size:10pt}.hero-meta{color:#617486;margin-top:12pt;font-size:8pt}.metrics{gap:10pt;margin:14pt 0}.metric{padding:10pt;border-color:#d6dfe7}.metric .value{font-size:17pt}.metric .label{font-size:8pt}.report{border:0;box-shadow:none;padding:0}.report p,.report li{font-size:10pt}.intro{font-size:10pt;padding-bottom:14pt}.key-finding{background:#f1f5f8!important;padding:12pt 14pt;break-inside:avoid}.report-section{padding:8pt 0 14pt;scroll-margin-top:0}h2{font-size:15pt;break-after:avoid;margin:16pt 0 12pt}h3{font-size:11.5pt;break-after:avoid;margin:16pt 0 8pt}.table-scroll{overflow:visible;border-radius:0;margin:14pt 0}table{font-size:8pt;min-width:100%;width:100%;table-layout:auto}th,td{padding:6pt 7pt;min-width:0!important;overflow-wrap:anywhere}thead{display:table-header-group}tr{break-inside:avoid}.status{font-size:7.5pt;padding:2pt 4pt}.equation{font-size:13pt;overflow:visible;padding:12pt;break-inside:avoid}.equation-line{white-space:normal}figure{break-inside:avoid;padding:7pt;margin:14pt 0}figcaption{font-size:8pt}.report-footer{font-size:8pt}.source-section{font-size:8pt}a{color:#245b80;text-decoration:none}a[href^="data:"]{text-decoration:underline}code{font-size:8pt}.report strong{color:inherit}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
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
const dialog = document.getElementById('figure-dialog');
const mainFigure = document.querySelector('figure img');
document.querySelector('.zoom-button')?.addEventListener('click', () => {
  dialog.querySelector('img').src = mainFigure.src;
  dialog.showModal();
});
document.querySelector('.close-button').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
"""


def data_uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def export():
    source = SOURCE.read_text(encoding="utf-8")
    formulas = [
        '<div class="equation" role="math" aria-label="视觉变化等于图像端点特征之差，文本变化等于文本端点特征之差">'
        '<span class="equation-line"><var>Δ<sub>v</sub></var> = <var>z<sub>v</sub></var>(I<sub>1</sub>) − <var>z<sub>v</sub></var>(I<sub>0</sub>)</span>'
        '<span class="equation-line"><var>Δ<sub>t</sub></var> = <var>z<sub>t</sub></var>(T<sub>1</sub>) − <var>z<sub>t</sub></var>(T<sub>0</sub>)</span></div>',
        '<div class="equation" role="math" aria-label="视觉变化与文本变化的点积等于正确配对分数和减去交叉配对分数和">'
        '<span class="equation-line"><var>Δ<sub>v</sub></var> · <var>Δ<sub>t</sub></var></span>'
        '<span class="equation-line">= s(I<sub>1</sub>, T<sub>1</sub>) + s(I<sub>0</sub>, T<sub>0</sub>)</span>'
        '<span class="equation-line indent">− s(I<sub>1</sub>, T<sub>0</sub>) − s(I<sub>0</sub>, T<sub>1</sub>)</span></div>',
    ]
    index = 0

    def replace_formula(match):
        nonlocal index
        replacement = formulas[index]
        index += 1
        return replacement

    source = re.sub(r"\\\[\s*.*?\s*\\\]", replace_formula, source, flags=re.S)
    assert index == len(formulas), "All display formulas must be converted for offline reading."
    soup = BeautifulSoup(markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"]), "html.parser")
    soup.h1.decompose()
    assert soup.find("p").get_text().startswith("日期：")
    soup.find("p").decompose()

    embedded = []
    for image in soup.find_all("img"):
        path = (SOURCE.parent / image["src"]).resolve()
        image["src"] = data_uri(path)
        image["width"] = "2550"
        image["height"] = "833"
        embedded.append(path.name)
        image_parent = image.parent
        caption = image_parent.find_next_sibling("p")
        figure = soup.new_tag("figure")
        image.extract()
        figure.append(image)
        figure.append(BeautifulSoup('<div class="figure-tools"><span>E1 / E2 · 三种子汇总</span><button class="zoom-button" type="button" aria-haspopup="dialog" aria-controls="figure-dialog">查看大图 ↗</button></div>', "html.parser"))
        if caption:
            figcaption = soup.new_tag("figcaption")
            for child in list(caption.contents):
                figcaption.append(child.extract())
            figure.append(figcaption)
            caption.decompose()
        image_parent.replace_with(figure)

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith(("http:", "https:", "#", "data:")):
            continue
        path = (SOURCE.parent / href).resolve()
        if path.is_file() and path.is_relative_to(ROOT):
            link["href"] = data_uri(path)
            link["download"] = path.name
            link["title"] = f"下载随文嵌入材料：{path.name}"
            embedded.append(path.name)

    tables = soup.find_all("table")
    for i, table in enumerate(tables, 1):
        wrapper = soup.new_tag("div", attrs={"class": "table-scroll", "tabindex": "0", "role": "region", "aria-label": f"数据表 {i}，可横向滚动"})
        table.wrap(wrapper)
        if i == 1:
            for row in table.select("tbody tr"):
                if row.find("td").get_text() in ("关系", "绑定"):
                    row["class"] = "structural-row"
        if i == len(tables):
            table["class"] = "evidence-table"
            for row in table.select("tbody tr"):
                status_cell = row.find_all("td")[1]
                value = status_cell.get_text()
                kind = "strong" if value == "强" else "pending" if value == "未测" else "limited"
                status_cell.clear()
                tag = soup.new_tag("span", attrs={"class": f"status {kind}"})
                tag.string = value
                status_cell.append(tag)

    intro = soup.new_tag("div", attrs={"class": "intro"})
    section = None
    sections = []
    for node in list(soup.contents):
        if getattr(node, "name", None) == "h2":
            section = soup.new_tag("section", attrs={"class": "report-section", "id": f"section-{len(sections)+1}"})
            sections.append(section)
        target = section if section is not None else intro
        target.append(node.extract())
    intro_paragraphs = intro.find_all("p", recursive=False)
    intro_paragraphs[1]["class"] = "key-finding"
    for section in sections:
        section.h2["id"] = section["id"] + "-heading"
        section["aria-labelledby"] = section.h2["id"]
    sections[-1]["class"] += " source-section"

    short_titles = ["数字口径与分析范围", "E1 · 概念与结构的落差", "大小题的无图像基线", "关系与位置的解释边界", "E2 · 背景转换的重复性", "幅度、方向与编辑区分", "环境转换的样本覆盖", "E3 · 形状与颜色的交互", "研究证据与机制链条", "下一阶段的最小验证", "复核与复现材料"]
    assert len(short_titles) == len(sections) == 11
    toc = "".join(f'<a href="#section-{i}"><span class="num">{i:02}</span><span>{html.escape(title)}</span></a>' for i, title in enumerate(short_titles, 1))
    article = str(intro) + "\n" + "\n".join(str(section) for section in sections)
    source_download = data_uri(SOURCE)
    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="description" content="GraphDelta 三模型、三种子预实验深化分析，包含 E1/E2/E3 证据复核、替代解释、机制边界与后续受控编辑验证。">
<title>GraphDelta｜预实验结论深化分析 · 2026-10-28</title>
<style>{CSS}</style>
</head>
<body id="top">
<div class="reading-progress" aria-hidden="true"></div>
<div class="shell">
<aside class="sidebar" aria-label="报告目录">
  <a class="brand" href="#top">Graph<span>Delta</span></a>
  <div class="edition">预实验 · 深化分析</div>
  <details class="toc-details" open>
    <summary>阅读目录</summary>
    <nav class="toc" aria-label="章节导航">{toc}</nav>
  </details>
  <div class="sidebar-footer">分析日期 · 2026.10.28<br>冻结模型 / 零样本评估<br>图表与复核材料已嵌入<br><a href="{source_download}" download="{SOURCE.name}">下载 Markdown 分析稿</a></div>
</aside>
<main>
  <header class="hero">
    <p class="eyebrow">GRAPHDELTA RESEARCH NOTE · STAGE 01</p>
    <h1>预实验结论深化分析</h1>
    <p class="hero-desc">证据强度、替代解释与 GraphDelta 的下一步。<br>从整体准确率走向逐因素诊断，明确已有证据与待验证机制。</p>
    <div class="hero-meta"><span>2026 年 10 月 28 日</span><span>3 个模型 × 3 个 seed</span><span>E1 / E2 / E3 完整结果复核</span></div>
  </header>
  <div class="metrics" aria-label="关键发现">
    <div class="metric"><div class="value">18.27–20.49<span class="unit">pp</span></div><div class="label">三模型颜色与绑定的<br>平均准确率差距</div></div>
    <div class="metric"><div class="value">≤ 0.143<span class="unit">pp</span></div><div class="label">大小题成绩与始终选 small<br>无图像基线的最大差距</div></div>
    <div class="metric"><div class="value">88% / 32%</div><div class="label">L/14 的 cube / cylinder<br>换形题准确率约值</div></div>
  </div>
  <div class="toolbar"><span class="toolbar-note">完整分析 · 11 节 · 图表与公式离线可读</span><button class="print-button" id="print-report" type="button">打印 / 保存 PDF</button></div>
  <article class="report" aria-label="完整分析正文">{article}</article>
  <footer class="report-footer"><span>GraphDelta · 预实验结论深化分析 · 2026-10-28<br>正文来源：经复核的 Markdown 分析稿；图像与引用材料随文嵌入。</span><a href="#top">返回顶部 ↑</a></footer>
</main>
</div>
<dialog id="figure-dialog" aria-labelledby="dialog-title"><div class="dialog-top"><span id="dialog-title">E1 与 E2 的三种子汇总 · 原尺寸图</span><button class="close-button" type="button">关闭</button></div><img alt="E1 与 E2 的三种子汇总放大图"></dialog>
<script>{JS}</script>
</body>
</html>'''
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"HTML: {OUTPUT}")
    print(f"Sections: {len(sections)}; tables: {len(tables)}; equations: {index}; embedded files: {len(embedded)}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    export()
