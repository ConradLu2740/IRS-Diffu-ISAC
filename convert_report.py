#!/usr/bin/env python3
r"""TECH_REPORT.md → TECH_REPORT.tex（arXiv-ready，pdflatex 兼容）转换器 v2。

v2 修复（对抗性审查发现）：
- \href{\url{A}{B}} 双重包裹 → \href{url}{text} 单层
- Unicode（Σ π τ λ α β ≈ ± ° ² ³ … — – × ⁻⁵）全部转 LaTeX
- 公式行进入 equation 数学环境（H(f) = Σ_i a_i exp(−j2π f τ_i)）
- 章节标题去除人工编号（"1. Introduction" → "Introduction"）
- 参考文献仅保留 thebibliography（md 列表跳过）
- 跳过首个 # 大标题（title 已手写）
用法：python3 convert_report.py
"""
import re

MD = "TECH_REPORT.md"
TEX = "TECH_REPORT.tex"

UNI = [
    ("Σ", r"$\Sigma$"), ("π", r"$\pi$"), ("τ", r"$\tau$"), ("λ", r"$\lambda$"),
    ("α", r"$\alpha$"), ("β", r"$\beta$"), ("≈", r"$\approx$"), ("±", r"$\pm$"),
    ("°", r"$^\circ$"), ("²", r"$^2$"), ("³", r"$^3$"), ("…", r"\ldots"),
    ("×", r"$\times$"), ("→", r"$\rightarrow$"), ("—", "--"), ("–", "--"),
    ("−", "-"), ("⁻", "-"), ("⁵", r"$^{5}$"), ("⁴", r"$^{4}$"), ("⁰", r"$^{0}$"),
    ("₁", r"$_1$"), ("₂", r"$_2$"), ("ᵢ", r"$_i$"), ("·", r"$\cdot$"),
]

# 公式行特征：含 H(f)、τ、Σ、exp(−j2π 等
def is_math_line(s):
    t = s.strip()
    # 严格判定：独立公式行（以 H( 开头），避免整段句子被误判
    return t.startswith("H(f)") or t.startswith("H(")

def to_math(s):
    """Unicode 文本公式 → LaTeX 数学（equation 内容）。"""
    import re as _re
    m = s.replace("H(f)", r"H(f)")
    # exp(-j2π f τ_i) → \exp(-j2\pi f \tau_i)（先于 π/τ 替换，用 Unicode 字面匹配）
    m = _re.sub("exp\\((?:\u2212|-)j2\u03c0 f \u03c4_i", r"\\exp(-j2\\pi f \\tau_i", m)
    m = m.replace("Σ_i", r"\sum_i").replace("Σ", r"\sum")
    m = m.replace("τ_i", r"\tau_i").replace("τ", r"\tau")
    m = m.replace("a_i", r"a_i").replace("π", r"\pi")
    m = m.replace("−", "-").replace("·", r"\cdot")
    return m

def inline(s):
    s = s.replace("\\", r"\textbackslash{} ")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", s)
    s = re.sub(r"`([^`]+?)`", r"\\texttt{\1}", s)
    # 链接 [text](url) → \href{url}{text}（在裸 URL 之前）
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\\href{\2}{\1}", s)
    # 裸 URL → \url{url}
    s = re.sub(r"(?<![\w{])(https?://[^\s\)\}]+)", r"\\url{\1}", s)
    # Unicode
    # superscript combinations first: 10⁻⁴ → $10^{-4}$ (before single-char ⁻/⁴ replacements)
    # NOTE: \d does NOT match superscript digits (Unicode No, not Nd), use explicit map
    SUP = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
    s = re.sub(r"10⁻([⁴-⁹])", lambda m: f"$10^{{-{SUP[m.group(1)]}}}$", s)
    for a, b in UNI:
        s = s.replace(a, b)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    s = s.replace("_", r"\_")
    return s

def md_table_to_latex(rows):
    ncols = len(rows[0])
    out = ["\\begin{table}[ht]", "\\centering", "\\small",
           "\\begin{tabular}{" + "l" * ncols + "}"]
    for ri, row in enumerate(rows):
        cells = [inline(c.strip()) for c in row]
        out.append(" & ".join(cells) + " \\\\")
        if ri == 0:
            out.append("\\hline")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)

PRE = r"""% ============================================================
% IRS-Diffu-ISAC Technical Report v1.3 (auto-converted from TECH_REPORT.md)
% arXiv-ready. Compile: pdflatex TECH_REPORT.tex
% ============================================================
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{RIS-Aided Integrated Sensing and Communication Toward Space ISAC: A Physics-Grounded Open-Source Engineering System}
\author{Conrad Lu\\[2pt] \small School of Information Science and Engineering, Northeastern University, Shenyang, China\\ \small\url{https://github.com/ConradLu2740/IRS-Diffu-ISAC}}
\date{Version 1.3, August 12, 2026}
\begin{document}
\maketitle"""

BIB = r"""\begin{thebibliography}{35}
\bibitem{liu2022isac} F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, ``Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond,'' \emph{IEEE J. Sel. Areas Commun.}, vol.~40, no.~6, pp. 1728--1767, 2022.
\bibitem{zhang2021perceptive} A. Zhang, M. L. Rahman, X. Huang, Y. J. Guo, S. Chen, and R. W. Heath, ``Perceptive mobile networks: Cellular networks with radio vision via joint communication and radar sensing,'' \emph{IEEE Veh. Technol. Mag.}, vol.~16, no.~2, pp. 20--30, 2021.
\bibitem{wu2019ris} Q. Wu and R. Zhang, ``Intelligent reflecting surface enhanced wireless network via joint active and passive beamforming,'' \emph{IEEE Trans. Wireless Commun.}, vol.~18, no.~11, pp. 5394--5409, 2019.
\bibitem{huang2019ris} C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, ``Reconfigurable intelligent surfaces for energy efficiency in wireless communication,'' \emph{IEEE Trans. Wireless Commun.}, vol.~18, no.~8, pp. 4157--4170, 2019.
\bibitem{3gppntn} 3GPP, ``Solutions for NR to support non-terrestrial networks (NTN),'' TR 38.821, Release 16, 2020.
\bibitem{wymeersch2021} H. Wymeersch \emph{et al.}, ``Integration of communication and sensing in 6G: A joint industrial and academic perspective,'' in \emph{Proc. IEEE PIMRC}, 2021.
\bibitem{3gppisac} 3GPP, ``Study on integrated sensing and communication,'' TR 22.837, Release 19, 2023.
\bibitem{sionna2022} J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, ``Sionna: An open-source, GPU-accelerated library for simulation of wireless systems,'' in \emph{Proc. IEEE SPAWC}, 2022.
\bibitem{hoots1980} F. R. Hoots and R. L. Roehrich, ``Spacetrack report no. 3: Models for propagation of NORAD element sets,'' U.S. Air Force Aerospace Defense Command, 1980.
\bibitem{lion2022} Y. Luo \emph{et al.}, ``LION: Latent point diffusion models for 3D shape generation,'' \emph{NeurIPS}, 2022.
\bibitem{pvd2021} Z. Lyu \emph{et al.}, ``PVD: Point-voxel diffusion for 3D generative modeling,'' \emph{ICCV}, 2021.
\bibitem{pointe2022} A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, ``Point-E: A system for generating 3D point clouds from complex prompts,'' \emph{arXiv:2212.08751}, 2022.
\bibitem{rohling1983} H. Rohling, ``Radar CFAR thresholding in clutter and multiple-target situations,'' \emph{IEEE Trans. Aerosp. Electron. Syst.}, vol.~AES-19, no.~4, pp. 608--621, 1983.
\bibitem{schmidt1986} R. Schmidt, ``Multiple emitter location and signal parameter estimation,'' \emph{IEEE Trans. Antennas Propag.}, vol.~34, no.~3, pp. 276--280, 1986.
\bibitem{skolnik2001} M. I. Skolnik, \emph{Introduction to Radar Systems}, 3rd ed., McGraw-Hill, 2001.
\bibitem{irsdiffu} C. Z. Lu, ``IRS-Diffu-ISAC: RIS-aided ISAC via diffusion models for 3D point cloud reconstruction,'' GitHub repository, 2026, \url{https://github.com/ConradLu2740/IRS-Diffu-ISAC}.
\bibitem{3gppisacrel20} 3GPP, ``Study on NR integrated sensing and communication,'' TR 38.765, Release 20, 2026.
\bibitem{3gppisacrel19} 3GPP, ``Service requirements for integrated sensing and communication,'' TS 22.137, Release 19, 2025.
\bibitem{ieee80211bf} IEEE, ``IEEE Standard for Information Technology---Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications---Amendment: WLAN Sensing,'' IEEE 802.11bf-2025, 2025.
\bibitem{iturm2160} ITU-R, ``IMT-2030 framework: Overall objectives of the future development of IMT for 2030 and beyond,'' Recommendation ITU-R M.2160-0, 2023.
\bibitem{liu2025rel19isac} Y. Liu, Y. Zhang, J. Zhang, Y. Pei, C. Zhao, S. Luo, \emph{et al.}, ``A comprehensive survey of 3GPP Release 19 ISAC channel modeling: From empirical features to unified methodology and standardized simulator,'' arXiv:2512.03506, 2025.
\bibitem{jamshed2026ntn} M. A. Jamshed, R. Singh, M. M. Saad, \emph{et al.}, ``ISAC-enabled non-terrestrial networks for 6G: Design principles, standardization, performance tradeoffs, and use cases,'' arXiv:2604.11593, 2026.
\bibitem{yang2026beam} J. Yang, H. Lee, and J. Choi, ``Beam training for RIS-aided ISAC systems,'' arXiv:2607.24003, 2026.
\bibitem{gkekas2026binary} A. Gkekas, A. I. Papadopoulos, P. A. Pantazopoulos, A. Lalas, K. Votis, C. Liaskos, ``Geometry-informed optimization of binary RIS configurations for communication and sensing,'' arXiv:2608.04133, 2026.
\bibitem{umra2025block} A. Umra, K. Weinberger, A. Khaleel, G. Enzner, and A. Sezgin, ``Short blocks, fast sensing: Finite blocklength tradeoffs in RIS-assisted ISAC,'' arXiv:2511.02673, 2025.
\bibitem{isaccedm2025} M. Farzanullah, H. Zhang, A. B. Sediq, A. Afana, and M. Erol-Kantarci, ``Conditional denoising diffusion for ISAC enhanced channel estimation in cell-free 6G,'' arXiv:2506.06942, 2025 (IEEE PIMRC).
\bibitem{radiodiff2026} X. Wang, Z. Fang, N. Cheng, \emph{et al.}, ``RadioDiff-Inverse: Diffusion-enhanced Bayesian inverse estimation for ISAC radio map construction,'' \emph{IEEE Trans. Wireless Commun.}, 2026.
\bibitem{rald2025} R. Zhang, B. Zeng, S. Wang, F. Zhou, and W. Wang, ``RaLD: Generating high-resolution 3D radar point clouds with latent diffusion,'' arXiv:2511.07067, 2025.
\bibitem{raddiff2025} J. Kwok, H. Caesar, and A. Palffy, ``4D-RaDiff: Latent diffusion for 4D radar point cloud generation,'' arXiv:2512.14235, 2025.
\bibitem{diffsurv2025} N. C. Luong, N. D. Hai, D. V. Le, H. T. Nguyen, T.-H. Vu, T. Huynh-The, \emph{et al.}, ``Diffusion models for future networks and communications: A comprehensive survey,'' arXiv:2508.01586, 2025 (submitted to Proceedings of the IEEE).
\bibitem{nsadm2025} N. D. M. Quang, C. Liu, S. Li, \emph{et al.}, ``Diffusion model-enhanced environment reconstruction in ISAC,'' arXiv:2511.19044, 2025 (submitted to IEEE Wireless Communications Letters).
\bibitem{dai2026pcimg} X. Dai, Y. Gao, H. Jiang, X. Yuan, and X. Wang, ``Conditional diffusion-based point cloud imaging for UAV position and attitude sensing,'' arXiv:2603.29822, 2026.
\bibitem{dai2026august} X. Dai, Y. Gao, H. Jiang, X. Yuan, and X. Wang, ``Conditional generative learning enabled wireless UAV sensing and tracking via point cloud imaging,'' arXiv:2607.14778, 2026.
\bibitem{yang2026leoisac} H. Yang, X. Chen, and Q. Wang, ``Robust design of integrated sensing and communication in LEO satellite systems,'' arXiv:2607.12337, 2026.
\bibitem{mimo2026limits} P.-C. Chen, M.-C. Lee, and Y.-C. Huang, ``Fundamental limits of MIMO-OTFS and MIMO-OFDM in high-dynamics ISAC: An antenna array architecture perspective,'' arXiv:2607.20200, 2026.
\end{thebibliography}

\end{document}"""


def strip_section_number(text):
    """'1. Introduction' → 'Introduction'；'1.1 Background' → 'Background'。"""
    m = re.match(r"^\d+(\.\d+)*\.?\s+(.*)$", text.strip())
    return m.group(2) if m else text.strip()


def main():
    with open(MD, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = [PRE]
    in_code = False
    in_abstract = False
    in_refs = False
    in_list = None
    first_h1_skipped = False
    i = 0
    while i < len(lines):
        line = lines[i]
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
        # 摘要
        if line.strip() == "## Abstract":
            in_abstract = True
            out.append("\\begin{abstract}")
            i += 1
            continue
        if in_abstract and line.strip().startswith("## "):
            in_abstract = False
            out.append("\\end{abstract}")
        # 参考文献段：跳过 md 列表（thebibliography 手写）
        if line.strip() == "## References":
            in_refs = True
            i += 1
            continue
        if in_refs:
            if line.strip().startswith("## ") or line.strip().startswith("# "):
                in_refs = False
            else:
                i += 1
                continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                if not first_h1_skipped:
                    first_h1_skipped = True  # 跳过主标题（title 已手写）
                else:
                    out.append("\\section*{" + inline(strip_section_number(text)) + "}")
            elif level == 2:
                out.append("\\section{" + inline(strip_section_number(text)) + "}")
            elif level == 3:
                out.append("\\subsection{" + inline(strip_section_number(text)) + "}")
            else:
                out.append("\\subsubsection{" + inline(strip_section_number(text)) + "}")
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
                in_list = "itemize" if lm else "enumerate"
                out.append("\\begin{" + in_list + "}")
            out.append("\\item " + inline((lm or nm).group(2)))
            i += 1
            continue
        if in_list is not None:
            out.append("\\end{" + in_list + "}")
            in_list = None
        # 公式行
        if is_math_line(line):
            out.append("\\begin{equation}")
            out.append(to_math(line.strip()))
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
    out.append(BIB)
    with open(TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {TEX} ({len(out)} lines)")


if __name__ == "__main__":
    main()
