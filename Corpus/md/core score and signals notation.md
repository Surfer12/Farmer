Considering notation definitions
The user asked me to define all notations, particularly focusing on MCDA notation in a succinct way. I noticed that terms like UW and ES are mentioned in other documents, such as "cursor_integration_of_deepmind_materialv1ff.md," where they represent user work and external sources. I’ll explain that UW ≡ S and ES ≡ N, while avoiding confusion by noting these synonyms. I’ll provide a structured bullet list of definitions without running any tools, and I’ll include a status update and summary as per the guidelines.
Preparing a list of core variables
I’m keeping this short and preparing a list of core variables and terminology. The core variables include things like x for claim/option index, S, N, various thresholds (like τ), and the components for probabilities such as P(H|E) for evidence. 
For sets, I need to consider actions, feasible options, and constraints. Temporal variables will be defined, covering time elements and normalized weights. I’ll also include regions, sensitivity parameters, and provide synonyms that clarify terms in this context, especially around internal versus external signals.
Defining units and bounds
I’m going to present the units and bounds clearly. I'll write the constraints like S and N being in the range [0,1], and that α is also in [0,1]. For the other parameters, I’ll specify that λ_i must be greater than 0, R_* should be non-negative, and β must be greater than or equal to 1. 
The threshold τ will be in the range (0,1], and the weights w(t) will be non-negative with either a sum or integral of 1. I’ll also include formulas for threshold transfers and define “κ” while keeping it minimal.


### Core score and signals
- S(x) ∈ [0,1]: internal signal (in‑repo/tests/static analysis) for item x
- N(x) ∈ [0,1]: external/canonical signal (specs/official docs) for x
- α ∈ [0,1]: allocation weight toward S (1−α toward N)
- O(α) = α S + (1−α) N ∈ [0,1]: hybrid (affine) evidence blend
- R_a ≥ 0, R_v ≥ 0: authority and verifiability/version risks
- λ1>0, λ2>0: risk weights
- pen = exp(−[λ1 R_a + λ2 R_v]) ∈ (0,1]: multiplicative risk penalty
- P(H|E) ∈ [0,1]: base posterior for hypothesis H given evidence E
- β ≥ 1: calibrated uplift factor (for certified/canonical evidence)
- post = min{β · P(H|E), 1} ∈ (0,1]: capped posterior
- Ψ = min{β · O(α) · pen, 1} ∈ [0,1]: guardrailed confidence score (pointwise)

### Thresholds and transfer
- τ ∈ (0,1]: action threshold (gate)
- κ > 0: uniform scaling of β
- Sub‑cap region U(β) = {β · O(α) · pen < 1}
- Threshold transfer (sub‑cap): if β→κ·β, preserve accepts by τ′ = κ · τ

### Sensitivities (sub‑cap)
- ∂Ψ/∂α = (S − N) · β · pen
- ∂Ψ/∂R_a = −β · O(α) · λ1 · pen
- ∂Ψ/∂R_v = −β · O(α) · λ2 · pen
- Monotone: increasing N or decreasing risks raises Ψ; decreasing α raises Ψ when N>S

### Temporal extension (optional, normalized)
- t ∈ T (time/index); w(t) ≥ 0 with ∫_T w(t) dt = 1 (or ∑ w_t = 1)
- O_t = α(t) S(t) + (1−α(t)) N(t); pen_t, β(t), P_t analogous
- Instantaneous Ψ_t = O_t · pen_t · min{β(t)·P_t, 1}
- Averaged score Ψ̄ = ∫ w(t) Ψ_t dt (or ∑ w_t Ψ_t) ∈ [0,1]
- Soft‑cap aggregator: R_t = β(t) O_t pen_t; I = ∫ w(t) R_t dt; Ψ̄ = 1 − exp(−I)
- Threshold transfer under β→κ·β: averaged (sub‑cap) τ′ = κ·τ; soft‑cap τ′ = 1 − (1 − τ)^κ

### MCDA/rules (when selecting among discrete options)
- A: option set; C: hard constraints; F = {a∈A | constraints satisfied}
- c(a) = [Ψ(a), cost(a), time(a), ROI(a), …]: criteria vector
- w (MCDA weights), M(c,w): monotone aggregator in the Ψ coordinate
- Decision: pick argmax_{a∈F} M(c(a), w)

- Defined every symbol used: core score, risks, uplift, thresholds, temporal weights/aggregators, and MCDA/rules terms.

Defining MCDA Notations
I need to respond concisely about various notations, especially focusing on MCDA notation as the user requested. Core notations I’ll cover include the alternative set A, feasible set F, and constraints C, among others. I’ll explain criteria vector c(a), weights w_j, the utility U(a), and logical components like the outranking S(a,b) relevant to ELECTRE/Promethee. Additionally, I should clarify threshold τ, and temporal notation of weights w(t). I might also look at monotonicity and transformations in criteria mappings.
Organizing MCDA Notations
The user requested a concise yet comprehensive response focused on MCDA. I’ll create a bullet list that covers key aspects like scaling and normalization forms, including min-max scaling and z-scores. I should define standardization functions for benefits and costs. Outranking thresholds, such as preference, indifference, and veto, along with concordance and discordance functions, are essential too. I'll clarify constraints using hard and soft constraints while mentioning multi-objective optimization. I'll use inline formatting for mathematical expressions to keep it tidy.
Listing MCDA Components
I need to create a structured list highlighting key components of MCDA. It will include sets and indices like 
A
A, 
J
J, and 
K
K, as well as variables 
a
a and 
b
b. I’ll define constraints 
C
C and feasible sets 
F
F, along with raw scores 
x
j
(
a
)
x 
j
​
 (a) and their directions. 
Normalization methods, weight vectors, and aggregator functions like WSM and WPM will also be included. Additionally, I’ll cover outranking methods like PROMETHEE and how they interact with rule engines and coupling with Ψ.
Outlining Weighted Synergy in MCDA
I need to address weighted synergy by mapping governance weights 
w
ψ
w 
ψ
​
  and include uncertainty through intervals for robust MCDA. I'll define 
x
j
(
a
)
x 
j
​
 (a) within bounds 
[
x
‾
,
x
ˉ
]
[ 
x
​
 , 
x
ˉ
 ] and consider worst-case utility or stochastic approaches. 
Next, I’ll describe sensitivity as the gradient of WSM and note how ties are broken lexicographically by 
Ψ
Ψ or cost. I should also mention any temporal aggregators discussed earlier. 
Finally, I’ll include a concise summary in bullet points at the end.
### MCDA-focused notation (with Ψ integration)

- Sets and indices
  - \(A\): finite set of alternatives/options; elements \(a,b\in A\).
  - \(J=\{1,\dots,m\}\): index set of criteria.
  - \(K\): index set of constraints/rules.

- Data per alternative a
  - Raw criterion scores: \(x_j(a)\in\mathbb{R}\) for each \(j\in J\).
  - Criterion direction: \(\mathrm{dir}_j\in\{\text{benefit},\text{cost}\}\).
  - Confidence score: \(\Psi(a)\in[0,1]\) (the pointwise guardrailed score).
  - Optional costs/times/etc.: e.g., \(x_{\mathrm{cost}}(a)\), \(x_{\mathrm{time}}(a)\).

- Constraints and feasible set
  - Hard rules: predicates \(g_k(a)\le 0\) (or Booleans \(H_k(a)\in\{0,1\}\)) for \(k\in K\).
  - Feasible set: \(F=\{\,a\in A:\forall k,\ g_k(a)\le 0\,\}\).
  - Soft rules (optional): penalties \(\rho_k\ge 0\) when a soft predicate is violated.

- Normalization (per criterion)
  - Min–max for benefit: \(z_j(a)=\dfrac{x_j(a)-\min_{b\in F}x_j(b)}{\max_{b\in F}x_j(b)-\min_{b\in F}x_j(b)}\).
  - Min–max for cost: \(z_j(a)=\dfrac{\max_{b\in F}x_j(b)-x_j(a)}{\max_{b\in F}x_j(b)-\min_{b\in F}x_j(b)}\).
  - Scale: \(z_j(a)\in[0,1]\) for all \(a\in F\).

- Weights and weight simplex
  - Weights: \(w=(w_1,\dots,w_m)\), \(w_j\ge 0\), \(\sum_j w_j=1\).
  - Ψ as a criterion: define \(j=\psi\) with \(x_\psi(a)=\Psi(a)\), \(z_\psi(a)=\Psi(a)\) (already in \([0,1]\)).

- Common MCDA aggregators (monotone in Ψ recommended)
  - Weighted sum (WSM): \(U_{\mathrm{WSM}}(a)=\sum_{j\in J} w_j\, z_j(a)\).
  - Weighted product (WPM): \(U_{\mathrm{WPM}}(a)=\prod_{j\in J} z_j(a)^{\,w_j}\).
  - TOPSIS:
    - Weighted matrix \(V_{j}(a)=w_j z_j(a)\).
    - Ideal/worst: \(v_j^*=\max_a V_j(a)\) if benefit else \(\min_a V_j(a)\); \(v_j^-=\min_a V_j(a)\) if benefit else \(\max_a V_j(a)\).
    - Distances: \(D^+(a)=\sqrt{\sum_j(V_j(a)-v_j^*)^2}\), \(D^-(a)=\sqrt{\sum_j(V_j(a)-v_j^-)^2}\).
    - Closeness: \(\mathrm{CC}(a)=\dfrac{D^-(a)}{D^+(a)+D^-(a)}\).
  - AHP (for weights): pairwise matrix \(P\), principal eigenvector \(\hat w\), consistency ratio \(CR\).
  - Outranking (PROMETHEE‑like):
    - Preference functions \(P_j(a,b)\in[0,1]\) with thresholds \(q_j\) (indifference), \(p_j\) (preference), \(v_j\) (veto).
    - Flows: \(\phi^+(a)=\frac{1}{|F|-1}\sum_{b\neq a}\sum_j w_j P_j(a,b)\); \(\phi^-(a)=\frac{1}{|F|-1}\sum_{b\neq a}\sum_j w_j P_j(b,a)\); net \(\phi(a)=\phi^+(a)-\phi^-(a)\).

- Decision pipeline with Ψ
  - Gate by confidence: \(F_\tau=\{a\in F:\Psi(a)\ge\tau\}\).
  - Aggregate among \(F_\tau\) using a monotone MCDA \(M\) (e.g., WSM, TOPSIS), ensuring it is strictly increasing in the Ψ coordinate.
  - Selection: \(a^\star\in\arg\max_{a\in F_\tau} M(a)\).
  - Soft‑rule penalties (optional): \(U'(a)=M(a)-\sum_{k}\rho_k\,\mathbf{1}[\text{violation}_k(a)]\).

- Sensitivity (WSM)
  - Weight: \(\partial U_{\mathrm{WSM}}(a)/\partial w_j = z_j(a)\).
  - Criterion: \(\partial U_{\mathrm{WSM}}(a)/\partial x_j(a) = w_j\,\partial z_j/\partial x_j\).
  - Ψ channel: \(\partial U/\partial \Psi(a) = w_\psi\) (direct, since \(z_\psi=\Psi\)).

- Threshold transfer (compatibility with Ψ)
  - Sub‑cap uniform uplift \( \beta\to\kappa\beta \) yields \( \Psi \to \kappa \Psi \).
  - Preserve \(F_\tau\) via \( \tau'=\kappa\,\tau \) (TOPSIS/WSM remain order‑consistent if other criteria unchanged).

- Temporal extension (optional, normalized)
  - Weights over time \(w(t)\ge 0\), \(\int w(t)\,dt=1\) (or \(\sum w_t=1\)).
  - Instantaneous \(\Psi_t(a)\); averaged \( \overline{\Psi}(a)=\int w(t)\Psi_t(a)\,dt \in[0,1]\).
  - Use \(\overline{\Psi}\) as \(x_\psi(a)\) to keep MCDA monotone and bounded.

- Notation summary
  - \(A,F,K,J,a,b,x_j,z_j,w_j,\Psi,\tau,\mathrm{dir}_j,P_j,q_j,p_j,v_j,\phi^\pm,\phi,\mathrm{CC},D^\pm,v_j^{*},v_j^{-}\).

- Defined MCDA sets, constraints, normalization, weights, aggregators (WSM, WPM, TOPSIS, AHP, outranking), and how Ψ enters as a criterion with gating and threshold transfer.
- Included sensitivities and an optional normalized temporal average for Ψ.

### Definitions (MCDA + Ψ)
- A: finite set of alternatives; a,b ∈ A
- K: index set of hard constraints; g_k(a) ≤ 0 encodes feasibility
- F = {a ∈ A | ∀k, g_k(a) ≤ 0}: feasible set
- J = {1,…,m}: index set of criteria; dir_j ∈ {benefit, cost}
- Raw scores: x_j(a) ∈ ℝ
- Min–max normalization:
  - benefit: z_j(a) = (x_j(a) − min_b x_j(b)) / (max_b x_j(b) − min_b x_j(b))
  - cost:    z_j(a) = (max_b x_j(b) − x_j(a)) / (max_b x_j(b) − min_b x_j(b))
- Weights: w_j ≥ 0, ∑_j w_j = 1
- Ψ(a) ∈ [0,1]: guardrailed confidence criterion; we use z_ψ(a) = Ψ(a)
- Gate: F_τ = {a ∈ F | Ψ(a) ≥ τ}
- Aggregators:
  - WSM: U(a) = ∑_j w_j z_j(a)
  - WPM: U(a) = ∏_j z_j(a)^{w_j}
  - TOPSIS: V_j(a) = w_j z_j(a); v_j^*, v_j^- are ideal/worst; D^±(a) Euclidean distances; CC(a) = D^−/(D^+ + D^-)
  - AHP: pairwise matrix → eigenvector weights w (used in any aggregator)
  - Outranking (PROMETHEE‑like): preference P_j(a,b) ∈ [0,1], flows φ^±, net φ = φ^+ − φ^−

### Theorem 1 (Normalization: range and order)
Claim: For each criterion j, z_j(a) ∈ [0,1]. If dir_j = benefit, z_j preserves order of x_j; if cost, it reverses. [conf 99%]
- Step 1: Denominator is max−min ≥ 0; if 0, treat criterion as constant (drop). [98%]
- Step 2: Affine map sends min→0, max→1, and interval to [0,1]. [99%]
- Step 3: Monotonicity follows from positive slope (+ for benefit, − for cost). [99%]

### Theorem 2 (Monotonicity in Ψ for standard MCDA)
Assume Ψ enters as z_ψ(a)=Ψ(a). Then, holding all other criteria fixed:
- WSM: U increases strictly with Ψ if w_ψ>0. [conf 99%]
  - U(a)=…+w_ψ Ψ(a); ∂U/∂Ψ = w_ψ>0. [99%]
- WPM: U increases strictly with Ψ if w_ψ>0 and Ψ∈(0,1]. [98%]
  - ∂U/∂Ψ = U·(w_ψ/Ψ) > 0 for Ψ>0. [98%]
- TOPSIS: CC increases with Ψ if only Ψ varies (all others fixed). [92%]
  - Raising Ψ moves z_ψ toward v_ψ^*, away from v_ψ^-; D^−↓, D^+↑ or flat ⇒ CC↑. [92%]
- Outranking: If P_ψ(a,b) is non‑decreasing in (Ψ(a)−Ψ(b)), then φ^+(a) increases with Ψ(a), φ^−(a) decreases. [90%]
- AHP: affects weights, not monotonicity post‑weights. [95%]

### Theorem 3 (Gating correctness)
F_τ = {a ∈ F | Ψ(a) ≥ τ} is a well‑defined feasible subset. [conf 99%]
- Step 1: F is feasible by constraints. [99%]
- Step 2: Ψ∈[0,1] so τ∈(0,1] defines a measurable/filter subset. [99%]

### Theorem 4 (Threshold transfer under sub‑cap uplift)
If all alternatives’ Ψ are sub‑cap and β→κ·β uniformly (κ>0), then Ψ′(a)=κΨ(a). Setting τ′=κτ yields F_τ = F′_τ′. [conf 98%]
- Step 1: Sub‑cap: Ψ=β·O·pen; scaling β→κβ gives Ψ′=κΨ. [98%]
- Step 2: Ψ(a)≥τ ⇔ κΨ(a)≥κτ ⇔ Ψ′(a)≥τ′. Hence F_τ=F′_τ′. [98%]

### Theorem 5 (Ranking invariance after transfer)
With τ′=κτ as above and other criteria unchanged:
- WSM/WPM: Rankings by U remain unchanged when only Ψ is multiplied by κ and w_ψ is fixed, because U′ differs by adding/multiplying a positive constant factor on the Ψ term only; pairwise order in Ψ channel preserved. [conf 95%]
  - WSM: ΔU(a,b) = w_ψ[κΨ(a)−κΨ(b)] = κ w_ψ ΔΨ(a,b). [96%]
  - WPM: Ratio U′(a)/U′(b) = [Ψ(a)/Ψ(b)]^{κ w_ψ} · (other fixed). [94%]
- TOPSIS: If min–max normalization is applied per criterion, pre‑normalization scaling by κ cancels in z_ψ (both min and max scale), leaving CC unchanged; hence ranking preserved. [conf 92%]
- Outranking: If z_ψ used and P_ψ depends on z_ψ differences, pre‑normalization scaling cancels; otherwise, preserve by scaling P_ψ thresholds consistently. [conf 88%]

### Corollary (MCDA with Ψ is stable and auditable)
- Adding Ψ as a criterion with w_ψ>0 yields monotone impact on MCDA scores (Theorem 2). [95%]
- Gating preserves feasibility (Theorem 3) and is invariant to uniform uplift with matched τ (Theorem 4). [98%]
- Standard aggregators keep ranking consistent under the transfer (Theorem 5), given usual normalizations. [92%]

### Notes on edge cases
- Degenerate criterion (max=min): drop or set z_j=0.5 constant; no effect on order. [97%]
- WPM with Ψ=0 and w_ψ>0 yields U=0; treat Ψ=0 as absolute veto if intended, or use ε‑floor. [90%]
- TOPSIS non‑Euclidean metrics may need separate monotonicity checks. [85%]
- Outranking veto thresholds can override Ψ monotonicity by design (policy). [85%]

- You now have formal definitions + stepwise proofs (with confidence) for sets, constraints, normalization, weights, WSM/WPM/TOPSIS/AHP/outranking, Ψ gating, and threshold transfer.

Structuring theorem response
I need to handle the user's request about "weighted synergy by mapping governance weights." They provided a lot of detailed specifications and asked for proof with confidence percentages at each step. I should keep the response compact yet thorough, delivering theorem definitions and brief proofs with confidence markers while avoiding excessive verbosity. I'll structure this using bullet lists for clarity and stick to a purely textual format for the math-related content. It's a balancing act, but I think I can do it!
Defining robust MCDA concepts
I need to define several concepts, including mapping governance weights and incorporating uncertainty intervals for robust MCDA. I'll define worst-case utility and the sensitivity gradient of WSM while addressing tie-breaking using Ψ or cost and incorporating TOPSIS metrics. I should demonstrate that the worst-case WSM acts as a lower bound and is monotonic in Ψ while linking governance and criterion weights. Additionally, I’ll provide a theorem about robust WSM, including details on sensitivity and scaling thresholds within a robust framework. It’s quite a task, but I’m up for it!
Explaining robust TOPSIS concepts
I'm focused on the robust version of TOPSIS, noting that distances are monotonic and non-increasing for benefit. With intervals, worst-case closeness happens at extremes, and I need to show that the robust CC lower bound is achieved by adjusting benefit and cost criteria. I’ll include inequalities and discuss robust outranking with preference functions. Governance weights as intervals are anchored by policy, and I'll detail how synergy mapping works. Finally, I want to prove that normalization preserves ordering monotonicity. It's a detailed process, but I think it can come together!
Structuring the theorem document
I need to create a concise, structured document that outlines several key theorems with stepwise proofs and confidence percentages. The user asked for clear definitions and to include tags like [conf xx%] for each step. It's a delicate balance to maintain minimal verbosity while ensuring all necessary information is present. I plan to address the robust WSM lower bound, sensitivity gradient, threshold transfer for robust gating, and more. Each theorem will have 2-4 proof steps, making it intricate yet manageable!
Crafting structured definitions and theorems
I'm laying out several key concepts related to governance weights and uncertainty intervals, highlighting how governance weights are defined from a governance vector. I need to present stepwise proofs for each theorem, including the robust WSM lower bound, sensitivity gradients, and tie-breaking methods, all while assigning confidence percentages. Additionally, I’ll cover robust TOPSIS concepts and threshold transfer scaling. I aim to keep the message minimal yet thorough and detailed enough, given its proof-heavy nature. Let's make sure to also mention synergy mapping!

### Definitions (governance → MCDA, robust intervals)
- Alternatives: A; a,b ∈ A. Constraints: g_k(a) ≤ 0, k ∈ K. Feasible set: F = {a ∈ A | ∀k, g_k(a) ≤ 0}.
- Criteria: J = {1,…,m}; raw scores x_j(a). Direction dir_j ∈ {benefit, cost}.
- Normalization (min–max on F): benefit z_j(a) = (x_j−min x_j)/(max x_j−min x_j); cost z_j(a) = (max x_j−x_j)/(max x_j−min x_j). All z_j ∈ [0,1].
- Confidence criterion: z_ψ(a) = Ψ(a) ∈ [0,1].
- Governance-weight mapping (synergy): base weights w^0 on non‑Ψ criteria, w^0_j ≥ 0, ∑_j≠ψ w^0_j = 1; governance score g_ψ ∈ [0,1]; mixing rate η ∈ (0,1].
  - w_ψ = η·g_ψ; w_j = (1−η)·w^0_j (j≠ψ). Then w ∈ Δ^m (∑ w_j = 1).
- Robust intervals: for each (j,a), x_j(a) ∈ [x̲_j(a), x̄_j(a)] ⇒ z_j(a) ∈ [z̲_j(a), z̄_j(a)] after direction‑aware normalization; for Ψ: Ψ(a) ∈ [Ψ̲(a), Ψ̄(a)].

- Gating: F_τ = {a ∈ F | Ψ̲(a) ≥ τ}.
- Aggregators:
  - WSM: U(a) = ∑_j w_j z_j(a).
  - WPM: U(a) = ∏_j z_j(a)^{w_j}.
  - TOPSIS: V_j(a)=w_j z_j(a); v_j^*, v_j^- ideals; D^+(a)=√∑(V_j−v_j^*)², D^−=√∑(V_j−v_j^−)²; CC(a)=D^−/(D^+ + D^−).
  - AHP: pairwise P → eigenvector w (plug into any aggregator).
  - Outranking: P_j(a,b)∈[0,1] with thresholds (q_j,p_j,v_j); flows φ^±, net φ=φ^+−φ^−.

### Theorem 1 (Governance mapping yields valid weights; monotone in g_ψ) [conf 99%]
- Step 1: Nonnegativity: w_ψ = η g_ψ ≥ 0; w_j=(1−η)w^0_j ≥ 0. [99%]
- Step 2: Sum to one: ∑_j w_j = η g_ψ + (1−η)∑_j≠ψ w^0_j = η g_ψ + (1−η) = 1. [99%]
- Step 3: ∂w_ψ/∂g_ψ = η > 0 ⇒ w_ψ strictly increases with governance confidence g_ψ. [99%]

### Theorem 2 (Robust WSM/WPM lower bounds; monotone in Ψ̲) [conf 97%]
- Step 1: For intervals z_j(a)∈[z̲_j,z̄_j], worst‑case WSM utility is U_min(a)=∑_j w_j z̲_j(a) (adversary picks least favorable per criterion). [97%]
- Step 2: For WPM, U_min(a)=∏_j z̲_j(a)^{w_j}. [95%]
- Step 3: If Ψ enters as z_ψ with interval [Ψ̲,Ψ̄], then ∂U_min/∂Ψ̲ = w_ψ > 0 (WSM); for WPM, ∂U_min/∂Ψ̲ = U_min·(w_ψ/Ψ̲) > 0 for Ψ̲>0. [97%]

### Theorem 3 (WSM sensitivities; robust subgradients) [conf 99%]
- Step 1: Gradient in weights: ∂U/∂w_j = z_j(a). [99%]
- Step 2: Gradient in scores: ∂U/∂x_j = w_j·∂z_j/∂x_j (direction‑aware). [99%]
- Step 3: Robust subgradient at the lower bound uses z̲_j(a): ∂U_min/∂w_j = z̲_j(a). [98%]
- Tie‑break: Lexicographic (U_min, Ψ̲, −cost) induces a total preorder; breaks WSM ties deterministically. [99%]

### Theorem 4 (Gating correctness and transfer with uncertainty) [conf 97%]
- Step 1: Using Ψ̲ ensures safety: if Ψ̲(a) ≥ τ then Ψ(a) ≥ τ for all realizations in the interval. [97%]
- Step 2: Sub‑cap uniform uplift β→κβ implies Ψ̲→κΨ̲. Setting τ′=κτ gives {a: Ψ̲≥τ} = {a: κΨ̲≥κτ} = {a: Ψ̲′≥τ′}. [97%]

### Theorem 5 (Robust TOPSIS lower‑bound behavior) [conf 88%]
- Step 1: With z_j(a)∈[z̲_j,z̄_j], define robust weighted bounds V̲_j(a)=w_j z̲_j(a), V̄_j(a)=w_j z̄_j(a). [95%]
- Step 2: A conservative (lower) closeness bound for a is obtained by evaluating D^+, D^− with V_j(a)=V̲_j(a), and computing v_j^*, v_j^- from opponents’ favorable extremes (benefit: v_j^* = max_b V̄_j(b), v_j^- = min_b V̲_j(b); reversed for cost). [88%]
- Step 3: This yields CC_lb(a) ≤ CC_true(a) by monotonicity of D^−↑ when a is pushed down and opponents up, and of D^+↓ likewise; thus CC_true is bounded below by CC_lb. [85%]

### Theorem 6 (Outranking robust lower bound) [conf 88%]
- Step 1: For each pair (a,b), worst‑case preference on a is P_j^lb(a,b)=P_j(Δz_j^lb) with Δz_j^lb = z̲_j(a)−z̄_j(b), thresholded to [0,1]. [90%]
- Step 2: φ_lb^+(a) = (1/(|F|−1))∑_{b≠a}∑_j w_j P_j^lb(a,b), φ_ub^−(a) = (1/(|F|−1))∑_{b≠a}∑_j w_j P_j^ub(b,a); then φ_lb(a)=φ_lb^+−φ_ub^− is a valid lower bound. [88%]

### Theorem 7 (TOPSIS and WSM ranking stability under threshold transfer) [conf 92%]
- Step 1 (WSM): Only the Ψ channel scales by κ; pairwise differences scale by κw_ψ ⇒ sign preserved; rankings by U_min unchanged if other criteria fixed. [96%]
- Step 2 (TOPSIS): With per‑criterion normalization, pre‑normalized κ cancels in z_ψ, so V_j and CC are unchanged; if κ only applies post‑normalization to Ψ, adjust z_ψ and recompute—order preserved if other criteria fixed and monotone. [88%]

### What to compute (robust, auditable)
- Gate: keep F_τ via Ψ̲ and τ (or τ′=κτ after uplift).
- WSM robust selection: U_min(a)=∑ w_j z̲_j(a); pick max; tie‑break by Ψ̲ then lower cost.
- TOPSIS robust: compute CC_lb(a) with a at V̲ and opponents at favorable extremes; pick max CC_lb.
- Outranking robust: compute φ_lb(a); pick max.
- Sensitivity: report ∂U_min/∂w_j = z̲_j(a), ∂U_min/∂Ψ̲ = w_ψ.

- Governance weights map cleanly to w (valid simplex); robust intervals yield safe lower bounds; gating and threshold transfer remain valid; gradients are simple and interpretable.
Clarifying chain-of-thought policy


I need to be concise, according to the verbosity level of 1. The user is asking why I have this information and why I would need to explain it. While I shouldn't share my internal reasoning, I can provide verifiable steps and definitions instead. So, I can clarify that I don't reveal private thoughts but can offer a summary and final results as part of the requested formal specification for transparency.
Explaining model rationale
I can clarify what I know: since you asked for model rationale, proofs, and MCDA integration, I've drawn from your files and standard sources. I explain this for auditability and reproducibility — definitions and theorems are tangible artifacts rather than my internal thought process. It's important to note that I won't share any private reasoning; I'll focus on providing results, proofs, or summaries as needed. If you prefer concise outputs, I can certainly do that!

Defining public methods
I think the user is asking for a definition of "public methods," which could refer to methods that are publicly verifiable or standard within a framework. They might want to know about public methods in the context of an API for the Ψ framework. I'll give a concise definition that includes programming aspects, like a function exposed by a class intended for external callers, ensuring input-output guarantees and stability. I should also mention key specifications like signature, preconditions, and thread-safety.
Defining public methods in the framework
I need to define "public methods" as "externally auditable procedures," particularly concerning the Ψ+MCDA pipeline. It would help to list typical public methods like computePsi, thresholdTransfer, normalizeCriteria, and others, along with their inputs, outputs, and determinism while noting there's no side-effects and outlining their complexity. Since the user's request likely aims to enumerate these methods in API format, I can present a simple API spec using markdown headings and bullet lists without heavy formatting. I'll keep it concise and clear!
Defining public methods
I'll outline several public methods with their respective parameters and outputs. For example, computePsi takes several inputs like S, N, and alpha to produce Psi. I can also detail methods like thresholdTransfer, normalizeCriterion, and gateByPsi, sharing their purposes and what they return.
Additionally, I’ll define methods like wsmScore, topsisScore, and how they relate to weights and criteria, while addressing their invariants and complexity. This will create a comprehensive API reference for users!
Presenting public methods concisely
I need to create a bullet list for public methods, keeping it concise with typed signature lines in code fences. First, I'll clarify that public methods are stable, documented interfaces that don’t expose internal workings; they should be deterministic and idempotent. Even though we aren't using tools at the moment, I should still provide a brief "Status update" and "Summary" to maintain structure and clarity in my communication. Let's craft that list and explanations effectively!
### Public methods (API) for Ψ + MCDA

- computePsi
  - Signature: computePsi(S, N, alpha, Ra, Rv, lambda1, lambda2, beta) -> {psi, O, pen, post}
  - Contract: 0 ≤ psi ≤ 1; O ∈ [0,1]; pen ∈ (0,1]; post ∈ (0,1].
  - Deterministic, pure.

- computePsiTemporal
  - Signature: computePsiTemporal(w, timeSeries, aggregator) -> psiBar
  - Inputs: w normalized weights over t; timeSeries provides S(t), N(t), alpha(t), Ra(t), Rv(t), beta(t); aggregator∈{mean, softcap}
  - Contracts: mean: psiBar ∈ [0,1]; softcap: psiBar ∈ [0,1).

- thresholdTransfer
  - Signature: thresholdTransfer(kappa, tau, mode) -> tauPrime
  - mode=subcap: tauPrime = kappa·tau; mode=softcap: tauPrime = 1 − (1 − tau)^kappa.

- normalizeCriterion
  - Signature: normalizeCriterion(values, direction) -> z in [0,1]
  - direction∈{benefit,cost}; min–max on feasible set.

- mapGovernanceWeights
  - Signature: mapGovernanceWeights(baseWeights, gPsi, eta) -> w
  - w_psi = eta·gPsi; w_j = (1−eta)·baseWeights_j; returns simplex weights (sum=1).

- gateByPsi
  - Signature: gateByPsi(alternatives, psiLower, tau) -> feasibleAlt
  - Uses lower bounds psiLower(a) to ensure safe gating.

- wsmScore / wsmScoreRobust
  - Signature: wsmScore(w, z) -> U; wsmScoreRobust(w, zLower) -> Umin
  - U = Σ w_j z_j; Umin = Σ w_j zLower_j.

- wpmScore / wpmScoreRobust
  - Signature: wpmScore(w, z) -> U; wpmScoreRobust(w, zLower) -> Umin
  - U = Π z_j^{w_j}; Umin = Π zLower_j^{w_j}.

- topsisScore / topsisScoreRobust
  - Signature: topsisScore(w, z, directions) -> CC; topsisScoreRobust(w, zLower, zUpper, directions) -> CCLower
  - Implements weighted matrix, ideal/worst, D±, closeness.

- ahpWeights
  - Signature: ahpWeights(pairwiseMatrix) -> {w, CR}
  - Returns eigenvector weights and consistency ratio.

- outrankingFlows / outrankingFlowsRobust
  - Signature: outrankingFlows(Pj, w) -> {phiPlus, phiMinus, phiNet}
  - Robust: uses worst/best deltas from zLower/zUpper to bound flows.

- zBoundsFromXBounds
  - Signature: zBoundsFromXBounds(xLower, xUpper, direction) -> {zLower, zUpper}.

- sensitivities
  - Signature: gradWSM(w, z) -> {dU_dw=z, dU_dz=w}; gradPsi(S,N,alpha,Ra,Rv,lambda1,lambda2,beta) -> partials
  - Sub-cap: dPsi/dalpha=(S−N)·beta·pen; dPsi/dRa=−beta·O·lambda1·pen; dPsi/dRv=−beta·O·lambda2·pen.

- tieBreak
  - Signature: tieBreak(alternatives, keys=[Umin desc, psiLower desc, cost asc]) -> winner
  - Deterministic lexicographic ordering.

- auditTrail
  - Signature: auditTrail(event, payload) -> recordId
  - Logs inputs, outputs, and parameters for reproducibility.

- invariants
  - psi, psiBar ∈ [0,1]; z ∈ [0,1]; weights nonnegative and sum to 1; all methods deterministic and side‑effect free except auditTrail.

- complexity (n=|A|, m=#criteria)
  - WSM/WPM: O(n·m)
  - TOPSIS: O(n·m)
  - Outranking: O(n^2·m)
  - AHP (weights): O(m^3) eigensolve

- error handling
  - Degenerate normalization (max=min): mark criterion constant or drop; WPM floor small z by ε>0 if needed.

- stability guarantees
  - Monotone in Ψ for WSM/WPM/TOPSIS (others fixed).
  - Threshold transfer preserves gating under sub‑cap/soft‑cap mappings.

  - It writes state: auditTrail persists a record (inputs, params, outputs, timestamps) to storage/telemetry. That external write is an observable change, so it’s not side-effect free.

- Why that’s necessary: you need durable provenance for reproducibility, compliance, and post-hoc review; returning data alone wouldn’t create an immutable trail.

- How to contain the side effect:
  - Inject a logger/sink interface; use fakes in tests.
  - Idempotency key to avoid duplicates; append-only storage.
  - Redact sensitive fields; include schema/version and clock source.
  - Support dryRun=true to no-op in scenarios that require purity.