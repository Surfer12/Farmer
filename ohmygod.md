Quick status: Providing the LaTeX document as a markdown code block with readable formatting. No blockers.


```LaTeX

\documentclass[11pt]{article}

\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{xcolor}

\geometry{margin=1in}

% Theorem environments
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}

% Confidence annotation macro
\newcommand{\conf}[2]{\textcolor{gray}{\footnotesize [confidence = #1;\ #2]}}

% Notation macros
\newcommand{\Ssig}{\mathsf{S}}
\newcommand{\Nsig}{\mathsf{N}}
\newcommand{\alloc}{\alpha}
\newcommand{\riskvec}{\mathbf{r}}
\newcommand{\rA}{R_a}
\newcommand{\rV}{R_v}
\newcommand{\lA}{\lambda_1}
\newcommand{\lV}{\lambda_2}
\newcommand{\uplift}{\beta}
\newcommand{\pen}{\mathrm{pen}}
\newcommand{\blend}{O}
\newcommand{\score}{\Psi}
\newcommand{\gscore}{\Phi}

\title{Equivalence and Divergence of the \texorpdfstring{$\score$}{Psi} Framework}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Notation}
\begin{itemize}[leftmargin=1.4em]
  \item $\Ssig,\Nsig\in[0,1]$: internal and canonical signal strengths.
  \item $\alloc\in[0,1]$: allocation between internal and canonical signals.
  \item $\riskvec=(\rA,\rV)\in[0,1]^2$: authority risk ($\rA$) and verifiability/version risk ($\rV$).
  \item $(\lA,\lV)$: positive risk weights.
  \item $\uplift\ge 1$: calibrated uplift factor with cap at 1.
  \item Affine blend: $\blend(\alloc)=\alloc\,\Ssig+(1-\alloc)\,\Nsig$.
  \item Exponential penalty: $\pen(\riskvec)=\exp\!\big(-[\lA\,\rA+\lV\,\rV]\big)$.
  \item Guardrailed score: $\score(\Ssig,\Nsig,\alloc,\riskvec,\lA,\lV,\uplift)=\min\{\uplift\cdot \blend(\alloc)\cdot \pen(\riskvec),\,1\}$.
  \item A second score with the same form and different parameter names is denoted
  $\gscore=\min\{\beta_g\cdot[\alpha_g \Ssig+(1-\alpha_g)\Nsig]\cdot \exp(-[\lambda_{1g}\rA+\lambda_{2g}\rV]),\,1\}$.
\end{itemize}

\section*{Axioms (structural assumptions)}
\begin{enumerate}[leftmargin=1.4em,label=A\arabic*:]
  \item Boundedness and monotonicity: outputs in $[0,1]$; increasing in evidence quality; decreasing in each risk.
  \item Canonical monotonicity: improving $\Nsig$ or reducing $(\rA,\rV)$ never decreases the score for any fixed $\alloc$.
  \item Allocation transparency: the evidence combiner is affine in $(\Ssig,\Nsig)$ with constant sensitivity in $\alloc$.
  \item Independent risks: joint risk discount multiplies for independent components and is continuous.
  \item Calibrated uplift with cap: a single factor $\uplift$ scales confidence, capped at 1, preserving order.
\end{enumerate}

\section*{Theorem 1 (Equivalence up to reparameterization)}
If $\gscore$ satisfies A1--A5 and has the same functional form as $\score$, then there exists a bijection of parameters
$(\alpha_g,\lambda_{1g},\lambda_{2g},\beta_g)\leftrightarrow(\alloc,\lA,\lV,\uplift)$ such that for all inputs,
$\gscore=\score$. Hence the frameworks are identical up to parameter names.

\paragraph{Proof}
\begin{enumerate}[leftmargin=1.6em]
  \item Define the parameter mapping $\alloc:=\alpha_g,\ \lA:=\lambda_{1g},\ \lV:=\lambda_{2g},\ \uplift:=\beta_g$.
  \conf{0.99}{direct identification of symbols}
  \item Under this mapping, the affine blends coincide:
  $\alpha_g \Ssig+(1-\alpha_g)\Nsig=\alloc\,\Ssig+(1-\alloc)\Nsig=\blend(\alloc)$.
  \conf{0.99}{algebraic identity}
  \item The penalties coincide:
  $\exp(-[\lambda_{1g}\rA+\lambda_{2g}\rV])=\exp(-[\lA\rA+\lV\rV])=\pen(\riskvec)$.
  \conf{0.99}{algebraic identity}
  \item The capped uplifts coincide:
  $\min\{\beta_g\,(\cdot),1\}=\min\{\uplift\,(\cdot),1\}$.
  \conf{0.99}{algebraic identity}
  \item Composing the identical subcomponents pointwise yields $\gscore=\score$ for all inputs.
  \conf{0.99}{function composition preserves equality}
\end{enumerate}
Thus, the two frameworks are equivalent up to the trivial reparameterization. \qed

\section*{Theorem 2 (Divergence beyond reparameterization)}
The scores $\score$ and $\gscore$ differ \emph{beyond trivial reparameterization} (i.e., no renaming aligns them pointwise) if and only if at least one of the following structural deviations holds:
\begin{itemize}[leftmargin=1.4em]
  \item[D1)] Non-affine evidence combiner (e.g., softmax/logistic) or non-constant sensitivity in $\alloc$ (violates A3).
  \item[D2)] Non-multiplicative or non-exponential risk aggregation (e.g., additive/clamped penalties) (violates A4).
  \item[D3)] Uplift without a hard cap at 1 or not order-preserving (violates A5).
\end{itemize}

\paragraph{Proof}
\textbf{(Only if).}
Assume $\score$ and $\gscore$ cannot be aligned by any bijection of $(\alloc,\lA,\lV,\uplift)\leftrightarrow(\alpha_g,\lambda_{1g},\lambda_{2g},\beta_g)$, yet both claim A1--A5.
\conf{0.85}{assumption setting; relies on precise notion of reparameterization}
If all three subcomponents (affine combiner, exponential penalty, capped uplift) match in form, Theorem 1 implies they \emph{can} be aligned; contradiction.
\conf{0.95}{contrapositive of Theorem 1}
Hence, at least one subcomponent must deviate in form, i.e., D1 or D2 or D3 holds.
\conf{0.95}{disjunction introduction from contradiction}

\smallskip
\textbf{(If).}
Suppose D1 holds. Then for some $(\Ssig,\Nsig,\alloc)$ the non-affine combiner induces a different ordering than the affine blend, so no parameter renaming makes the functions equal on all inputs.
\conf{0.90}{existence of counterexample via nonlinearity and order reversal}
Similarly, under D2, additive/clamped penalties cannot replicate multiplicative exponential behavior over independent risks, producing inputs where the outputs differ.
\conf{0.90}{distinct algebraic families imply pointwise disagreement}
Under D3, lack of a hard cap changes outputs near or above 1, again yielding pointwise differences.
\conf{0.92}{boundary behavior differs; cap enforces saturation}
Thus, any of D1--D3 implies divergence beyond reparameterization. \qed

\section*{Confidence scoring rubric}
We annotate each proof step with a bracketed score in $[0,1]$:
\begin{itemize}[leftmargin=1.4em]
  \item 0.95--1.00: algebraic identity/compositionality; widely accepted and fully checkable.
  \item 0.85--0.94: relies on standard reasoning (e.g., existence of counterexamples) with minor assumptions.
  \item 0.70--0.84: relies on modeling choices or informal definitions that could vary by convention.
\end{itemize}
Theorem 1 steps are algebraic/compositional, hence $\ge 0.99$. Theorem 2 ``only if'' uses a contrapositive of Theorem 1 ($0.95$) with an assumption framing (0.85). The ``if'' direction constructs counterexamples by family differences (0.90--0.92).

\end{document}
```









### References (all‑inclusive)  