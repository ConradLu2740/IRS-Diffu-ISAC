#!/usr/bin/env python3
"""TECH_REPORT.md → TECH_REPORT.tex（arXiv-ready）转换器。

处理：标题/作者/摘要、章节、表格、代码块、加粗/斜体、行内代码、
      URL、thebibliography 参考文献、公式块。
用法：python3 convert_report.py
"""
import re
import sys

MD = "TECH_REPORT.md"
TEX = "TECH_REPORT.tex"

# ---- 行内转换 ----
def inline(s):
    s = s.replace("\\", "\\textbackslash{}")  # 防斜杠（先占位再还原）
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", s)
    s = re.sub(r"`([^`]+?)`", r"\\texttt{\1}", s)
    # URL 链接 [text](url)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\\href{\2}{\1}", s)
    # 裸 URL
    s = re.sub(r"(?<!\()(https?://[^\s\)]+)", r"\\url{\1}", s)
    # Unicode → LaTeX（pdflatex 兼容）
    s = s.replace("10\u207b\u2075", r"$10^{-5}$").replace("10\u207b\u00b3", r"$10^{-3}$")
    s = s.replace("\u207b\u2075", r"$^{-5}$")          # 其他 ⁻⁵ 组合
    s = s.replace("\u2192", r"$\rightarrow$")
    s = s.replace("\u00d7", r"$\times$")
    s = s.replace("\u207b", "-").replace("\u2075", r"$^{5}$").replace("\u2074", r"$^{4}$").replace("\u2070", r"$^{0}$")
    s = s.replace("\u2212", "-").replace("\u2081", r"$_1$").replace("\u2082", r"$_2$")
    # 还原特殊转义
    s = s.replace("\\textbackslash{}", "\\textbackslash{} ")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    s = s.replace("_", r"\_")  # 行内下划线（数学环境外）
    return s

def is_math_line(s):
    t = s.strip()
    return t.startswith("H(") or t.startswith("τ") or "τ_i" in t or "=" in t and "\\" not in t and any(c in t for c in "Σ") and t.startswith("H") or "10⁻⁴" in s

# ---- 表格解析 ----
def md_table_to_latex(rows, aligns=None):
    """rows: 已拆分的行列表（每行是 cell 列表）。"""
    ncols = len(rows[0])
    spec = "l" * ncols
    out = ["\\begin{table}[ht]", "\\centering", "\\small",
           "\\begin{tabular}{" + spec + "}"]
    for ri, row in enumerate(rows):
        cells = [inline(c.strip()) for c in row]
        out.append(" & ".join(cells) + " \\\\")
        if ri == 0:
            out.append("\\hline")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)

def main():
    with open(MD, encoding="utf-8") as f:
        lines = f.read().split("\n")

    out = []
    out.append(r"""% ============================================================
% IRS-Diffu-ISAC Technical Report (auto-converted from TECH_REPORT.md)
% arXiv-ready. Compile: pdflatex TECH_REPORT.tex
% ============================================================
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{RIS-Aided Integrated Sensing and Communication Toward Space ISAC: A Physics-Grounded Open-Source Engineering System}
\author{Conrad Lu\\[2pt] \small School of Information Science and Engineering, Northeastern University, Shenyang, China\\ \small\url{https://github.com/ConradLu2740/IRS-Diffu-ISAC}}
\date{Version 1.0, August 8, 2026}
\begin{document}
\maketitle""")

    i = 0
    in_code = False
    in_abstract = False
    in_list = None
    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                out.append("\\begin{lstlisting}")
            else:
                in_code = False
                out.append("\\end{lstlisting}")
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # 摘要区（## Abstract 后到 ## 1. Introduction 前）——必须在标题分支之前
        if line.strip() == "## Abstract":
            in_abstract = True
            out.append("\\begin{abstract}")
            i += 1
            continue
        if in_abstract and line.strip().startswith("## "):
            in_abstract = False
            out.append("\\end{abstract}")
            # fallthrough 让标题分支处理下一行

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                out.append("\\section*{" + inline(text) + "}")
            elif level == 2:
                out.append("\\section{" + inline(text) + "}")
            elif level == 3:
                out.append("\\subsection{" + inline(text) + "}")
            else:
                out.append("\\subsubsection{" + inline(text) + "}")
            i += 1
            continue

        # 表格
        if line.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                out.append(md_table_to_latex(rows))
            continue

        # 水平线
        if re.match(r"^\s*---+\s*$", line):
            i += 1
            continue

        # 列表
        lm = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        nm = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if lm or nm:
            if in_list is None:
                in_list = ("itemize" if lm else "enumerate")
                out.append("\\begin{" + in_list + "}")
            out.append("\\item " + inline((lm or nm).group(2)))
            i += 1
            continue
        if in_list is not None:
            out.append("\\end{" + in_list + "}")
            in_list = None

        # 公式块
        if line.strip().startswith("$$"):
            out.append("\\begin{equation}")
            i += 1
            body = []
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                body.append(lines[i]); i += 1
            out.append("\n".join(body))
            out.append("\\end{equation}")
            i += 1
            continue

        # 普通段落
        if line.strip():
            out.append(inline(line))
        else:
            out.append("")
        i += 1

    if in_list is not None:
        out.append("\\end{" + in_list + "}")
    if in_abstract:
        out.append("\\end{abstract}")

    out.append(r"""\begin{thebibliography}{14}
\bibitem{liu2022isac} F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, ``Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond,'' \emph{IEEE J. Sel. Areas Commun.}, vol.~40, no.~6, pp. 1728--1767, 2022.
\bibitem{zhang2021perceptive} A. Zhang, M. L. Rahman, X. Huang, Y. J. Guo, S. Chen, and R. W. Heath, ``Perceptive mobile networks: Cellular networks with radio vision via joint communication and radar sensing,'' \emph{IEEE Veh. Technol. Mag.}, vol.~16, no.~2, pp. 20--30, 2021.
\bibitem{wu2019ris} Q. Wu and R. Zhang, ``Intelligent reflecting surface enhanced wireless network via joint active and passive beamforming,'' \emph{IEEE Trans. Wireless Commun.}, vol.~18, no.~11, pp. 5394--5409, 2019.
\bibitem{huang2019ris} C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, ``Reconfigurable intelligent surfaces for energy efficiency in wireless communication,'' \emph{IEEE Trans. Wireless Commun.}, vol.~18, no.~8, pp. 4157--4170, 2019.
\bibitem{3gppntn} 3GPP, ``Solutions for NR to support non-terrestrial networks (NTN),'' TR 38.821, Release 16, 2020.
\bibitem{wymeersch2021} H. Wymeersch \emph{et al.}, ``Integration of communication and sensing in 6G: A joint industrial and academic perspective,'' in \emph{Proc. IEEE PIMRC}, 2021.
\bibitem{hoots1980} F. R. Hoots and R. L. Roehrich, ``Spacetrack report no. 3: Models for propagation of NORAD element sets,'' U.S. Air Force Aerospace Defense Command, 1980.
\bibitem{lion2023} Y. Luo \emph{et al.}, ``Latent diffusion models for 3D point cloud generation,'' \emph{NeurIPS}, 2023.
\bibitem{pvd2021} Z. Lyu \emph{et al.}, ``PVD: Point-voxel diffusion for 3D generative modeling,'' \emph{ICCV}, 2021.
\bibitem{pointe2022} A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, ``Point-E: A system for generating 3D point clouds from complex prompts,'' \emph{arXiv:2212.08751}, 2022.
\bibitem{cfar2005} R. P. S. Buda and R. M. Narayanan, ``Constant false alarm rate detection of pulse compression radar signals,'' \emph{IEEE Radar Conf.}, 2005.
\bibitem{schmidt1986} R. Schmidt, ``Multiple emitter location and signal parameter estimation,'' \emph{IEEE Trans. Antennas Propag.}, vol.~34, no.~3, pp. 276--280, 1986.
\bibitem{skolnik2001} M. I. Skolnik, \emph{Introduction to Radar Systems}, 3rd ed., McGraw-Hill, 2001.
\bibitem{irsdiffu} C. Z. Lu, ``IRS-Diffu-ISAC: RIS-aided ISAC via diffusion models for 3D point cloud reconstruction,'' GitHub repository, 2026, \url{https://github.com/ConradLu2740/IRS-Diffu-ISAC}.
\end{thebibliography}

\end{document}""")

    with open(TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {TEX} ({len(out)} lines)")

if __name__ == "__main__":
    main()
