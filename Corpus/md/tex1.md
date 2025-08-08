\documentclass[11pt]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs,array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}

\hypersetup{
  pdftitle={UOIF: IMO 2024--2025 Dialectical Synthesis and \texorpdfstring{$\Psi(x)$}{Psi(x)} Reevaluation},
  pdfauthor={UOIF Working Note},
  pdfsubject={Unified Olympiad Information Framework (UOIF) for IMO 2024--2025},
  pdfkeywords={IMO, DeepMind, Evan Chen, AoPS, UOIF, scoring, reliability, dialectical synthesis}
}

\newcommand{\Sx}{S(x)}
\newcommand{\Nx}{N(x)}
\newcommand{\Px}{\Psi(x)}
\newcommand{\Pa}{P(H\mid E,\beta)}

\title{Unified Olympiad Information Framework (UOIF):\\
IMO 2024--2025 Dialectical Synthesis and $\Px$ Reevaluation}
\author{UOIF Working Note}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\textbf{Quick status.} Integrated DeepMind 2024 and 2025 solutions as independent expert/organization interpretive sources with high authority (2025: IMO-certified grading). Updated citations, roles, $s(c)$, $\alpha(t)$, $\Nx$, and $\Px$ per year. Raised 2025 $\Nx$ to 0.95 (validated solutions) and promoted to \emph{Empirically Grounded} for solved problems (5/6). Enhanced 2024 to $\Nx=0.96$ (DeepMind P1/P2/P4). Full primitives remain \emph{pending canonical} until official IMO artifacts are posted. Minor divergences (method variants) resolved by preferring certified DeepMind for scoring uplift while citing Evan/AoPS for coherence.
\end{abstract}

\section{Unified ruleset for IMO 2024--2025 analysis}

\subsection{Source roles and hierarchy}
\begin{itemize}[leftmargin=1.2em]
  \item \textbf{Canonical (pending where not yet posted):} official IMO problems/solutions, shortlist, jury PDFs (site: \href{https://imo-official.org}{imo-official.org}).
  \item \textbf{Historical mirror (context for primitives; not canonical for 2025 until posted):} \href{https://olympiads.win.tue.nl/imo}{olympiads.win.tue.nl/imo}.
  \item \textbf{Expert interpretive:}
    \begin{itemize}[leftmargin=1.2em]
      \item 2025: Evan Chen, ``IMO 2025 Solution Notes'' (PDF): \href{https://web.evanchen.cc/exams/IMO-2025-notes.pdf}{web.evanchen.cc/exams/IMO-2025-notes.pdf}
      \item 2024: Evan Chen site root (use year PDF when cited): \href{https://web.evanchen.cc/}{web.evanchen.cc}
      \item DeepMind solution sets (2024/2025): treat as expert interpretive; high authority for 2025 due to IMO-certified grading.
    \end{itemize}
  \item \textbf{Community interpretive (AoPS per-problem, 2025):}
    P1 \href{https://aops.com/community/p35332003}{(link)},
    P2 \href{https://aops.com/community/p35332018}{(link)},
    P3 \href{https://aops.com/community/p35332016}{(link)},
    P4 \href{https://aops.com/community/p35347364}{(link)},
    P5 \href{https://aops.com/community/p35341177}{(link)},
    P6 \href{https://aops.com/community/p35341197}{(link)}.
\end{itemize}
\textbf{Priority for primitives:} Canonical $>$ Expert (Evan/DeepMind) $>$ AoPS $>$ Mirror (context).

\subsection{Claim classes and gating}
\begin{itemize}[leftmargin=1.2em]
  \item \textbf{Primitive:} exact statements/answers/credits. 2025: cite Evan 2025 PDF + AoPS; mark ``pending official.'' 2024: cite Evan 2024 + established archives; include official text when available.
  \item \textbf{Interpretation:} proofs, methods, lemmas, comparisons. Require $\ge 1$ expert (Evan/DeepMind) or AoPS link per problem.
  \item \textbf{Speculative/context:} heuristics/naming/history. Label as speculative; do not promote.
\end{itemize}

\subsection{Scoring $s(c)$ and allocation $\alpha(t)$}
\[
s(c)=w_{\text{auth}}\cdot \mathrm{Authority}+w_{\text{ver}}\cdot \mathrm{Verifiability}+w_{\text{depth}}\cdot \mathrm{Depth}+w_{\text{align}}\cdot \mathrm{Intent}+w_{\text{rec}}\cdot \mathrm{Recency}-w_{\text{noise}}\cdot \mathrm{Noise}.
\]
\begin{center}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Class & $w_{\text{auth}}$ & $w_{\text{ver}}$ & $w_{\text{depth}}$ & $w_{\text{align}}$ & $w_{\text{rec}}$ & $w_{\text{noise}}$\\
\midrule
Primitives & 0.35 & 0.30 & 0.10 & 0.15 & 0.07 & 0.23\\
Interpretations & 0.20 & 0.20 & 0.25 & 0.25 & 0.05 & 0.15\\
\bottomrule
\end{tabular}
\end{center}
\textbf{Allocations.} 2025 primitives: $\alpha=0.15$--$0.20$ (favor external interpretive artifacts). 2024 primitives: $\alpha=0.10$--$0.15$ if canonical present; else similar to 2025. Interpretations (both years): $\alpha=0.35$--$0.45$.\\
\textbf{External reliability $\Nx$.} 2025 (DeepMind certified + Evan + AoPS): $\Nx\approx 0.95$ for solved problems; 2024 (DeepMind P1/P2/P4 + established archives): $\Nx\approx 0.96$.\\
\textbf{Posterior calibration.} $P(H\mid E)\in[0.85,0.90]$; $\beta=1.05$ pre-canonical uplift; raise to $1.10$--$1.20$ once canonical posted.

\subsection{Decision equation}
\begin{align*}
\Px &= \bigl[\alpha \cdot \Sx + (1-\alpha)\cdot \Nx\bigr]\cdot \exp\!\Bigl(-\bigl[\lambda_1 R_{\text{authority}}+\lambda_2 R_{\text{verifiability}}\bigr]\Bigr)\cdot \Pa,\\
\lambda_1&=0.85,\quad \lambda_2=0.15.
\end{align*}
Increase $R_{\text{authority}}$ if no canonical; increase $R_{\text{verifiability}}$ if missing direct artifact URLs.

\subsection{Coherence, order, and conflicts}
\textbf{Coherence checks:} (i) String: quoted primitives match Evan PDF (2025) or official text (2024) with low edit distance; (ii) Asset: include problem-specific AoPS link(s) and Evan PDF; add official archive links when available; (iii) Timestamp: use latest Evan notes; record AoPS post IDs.\\
\textbf{Non-commutative evidence order:} For primitives, consult expert PDFs (Evan/DeepMind) before AoPS; for interpretations, combine Evan + DeepMind, then AoPS.\\
\textbf{Conflict resolution:} Prefer Evan for edited coherence; prefer DeepMind for certified scoring uplift; present both when steps differ; mirror remains context until 2025 canonical is posted.

\section{Evaluation: IMO 2025 (DeepMind certified; 5/6 solved)}
\textbf{Inputs.} $\Sx=0.60$; $\Nx=0.95$; $\alpha\in[0.15,0.20]$; $R_{\text{authority}}=0.15$; $R_{\text{verifiability}}=0.05$; $\Pa=0.90\times 1.05=0.945$. Penalty $=\exp\bigl(-[0.85\cdot 0.15+0.15\cdot 0.05]\bigr)=\exp(-0.135)\approx 0.8737$.\\[0.3em]
\textbf{Hybrid.} $O_{\text{hybrid}}=\alpha\cdot 0.60 + (1-\alpha)\cdot 0.95 = 0.95-0.35\alpha$.\\
\textbf{Results.}
\begin{align*}
\alpha=0.15:&\quad O=0.8975,\quad \Px \approx 0.8975\times 0.8737\times 0.945 \approx 0.741,\\
\alpha=0.20:&\quad O=0.8800,\quad \Px \approx 0.8800\times 0.8737\times 0.945 \approx 0.726,\\
\alpha=0.17:&\quad O=0.8905,\quad \Px \approx 0.737\ \text{(recommended operating point).}
\end{align*}
\textbf{Label.} \emph{Empirically Grounded} for solved problems (5/6). The unsolved problem remains \emph{Interpretive/Contextual} pending canonical artifacts.\\[0.3em]
\textbf{Citations.} DeepMind blog (gold, certified): \href{https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/}{link}; DeepMind solutions PDF (all problems): \href{https://storage.googleapis.com/deepmind-media/gemini/IMO_2025.pdf}{link}; Evan 2025 PDF: \href{https://web.evanchen.cc/exams/IMO-2025-notes.pdf}{link}; AoPS threads P1--P6: \href{https://aops.com/community/p35332003}{P1}, \href{https://aops.com/community/p35332018}{P2}, \href{https://aops.com/community/p35332016}{P3}, \href{https://aops.com/community/p35347364}{P4}, \href{https://aops.com/community/p35341177}{P5}, \href{https://aops.com/community/p35341197}{P6}.

\section{Evaluation: IMO 2024 (DeepMind P1/P2/P4; enhanced)}
\textbf{Inputs.} $\Sx=0.60$; $\Nx=0.96$; $\alpha\in[0.10,0.15]$; $R_{\text{authority}}=0.10$; $R_{\text{verifiability}}=0.05$; $\Pa=0.90\times 1.05=0.945$. Penalty $=\exp\bigl(-[0.85\cdot 0.10+0.15\cdot 0.05]\bigr)=\exp(-0.0925)\approx 0.9117$.\\[0.3em]
\textbf{Hybrid.} $O_{\text{hybrid}}=0.96-0.36\alpha$.\\
\textbf{Results.}
\begin{align*}
\alpha=0.10:&\quad O=0.9240,\quad \Px \approx 0.9240\times 0.9117\times 0.945 \approx 0.796,\\
\alpha=0.15:&\quad O=0.9060,\quad \Px \approx 0.9060\times 0.9117\times 0.945 \approx 0.781.
\end{align*}
\textbf{Label.} \emph{Primitive/Empirically Grounded} (supported by DeepMind P1/P2/P4 and established archives; further uplift upon full canonical swap).\\[0.3em]
\textbf{Citations.} DeepMind 2024 blog: \href{https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/}{link}; 2024 solution pages: \href{https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/index.html}{Index}, \href{https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P1/index.html}{P1}, \href{https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P2/index.html}{P2}, \href{https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P4/index.html}{P4}; Evan site root: \href{https://web.evanchen.cc/}{link}; Mirror (context): \href{https://olympiads.win.tue.nl/imo}{link}.

\section{Promotion triggers and expected post-canonical gains}
On official IMO artifacts going live:
\begin{itemize}[leftmargin=1.2em]
  \item Reduce $\alpha \to 0.10$--$0.12$, raise $\Nx \to 0.97$--$0.98$, increase $\beta \to 1.10$--$1.20$, and lower $R_{\text{authority}}$.
  \item \textbf{Expected:} 2025 full Primitive with $\Px \approx 0.87$--$0.90$; 2024 $\Px \approx 0.83$--$0.86$.
\end{itemize}

\section{Summary table}
\begin{center}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Year & Scope & $\alpha$ & $\Nx$ & Penalty & $\Pa$ & $\Px$ (range)\\
\midrule
2025 & 5/6 solved & $0.15$--$0.20$ & $0.95$ & $0.8737$ & $0.945$ & $0.726$--$0.741$\\
2024 & P1/P2/P4 & $0.10$--$0.15$ & $0.96$ & $0.9117$ & $0.945$ & $0.781$--$0.796$\\
\bottomrule
\end{tabular}
\end{center}

\section*{Notes on presentation and logging}
Label each claim as [Primitive]/[Interpretation]/[Speculative]; attach links and one-line rationale. Maintain a changelog for canonical link swaps; recompute $\Px$ after each swap and update labels. Persisting conflicts: present both and mark ``unresolved pending official.''

\end{document}