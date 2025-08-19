import math
from typing import Iterable, List, Tuple, Union, Optional

Number = Union[float, int]
ArrayLike = Union[Number, List[Number], Tuple[Number, ...]]


def _to_list(x: ArrayLike) -> List[float]:
	if isinstance(x, (list, tuple)):
		return [float(v) for v in x]
	return [float(x)]


def _broadcast_to_len(values: List[float], length: int) -> List[float]:
	if len(values) == length:
		return values
	if len(values) == 1 and length > 1:
		return [values[0] for _ in range(length)]
	raise ValueError("Inputs must have the same length or be scalar for broadcasting.")


def _align_shapes(*arrays: List[float]) -> List[List[float]]:
	max_len = max(len(a) for a in arrays)
	return [_broadcast_to_len(a, max_len) for a in arrays]


def _clip(x: float, lo: float, hi: float) -> float:
	return min(max(x, lo), hi)


def sigmoid(z: Number) -> float:
	return 1.0 / (1.0 + math.exp(-float(z)))


def logit(p: Number) -> float:
	p_clipped = _clip(float(p), 1e-12, 1.0 - 1e-12)
	return math.log(p_clipped / (1.0 - p_clipped))


def calibrate_probability_logit_shift(p: ArrayLike, beta: float, clip: bool = True) -> List[float]:
	p_list = _to_list(p)
	out: List[float] = []
	for val in p_list:
		z = logit(val) + math.log(beta)
		p_adj = sigmoid(z)
		out.append(_clip(p_adj, 0.0, 1.0) if clip else p_adj)
	return out


def calibrate_probability_multiplicative(p: ArrayLike, beta: float, clip: bool = True) -> List[float]:
	p_list = _to_list(p)
	out: List[float] = []
	for val in p_list:
		p_adj = float(val) * beta
		out.append(_clip(p_adj, 0.0, 1.0) if clip else p_adj)
	return out


def compute_psi(
	S: ArrayLike,
	N: ArrayLike,
	alpha: ArrayLike,
	Rcog: ArrayLike,
	Reff: ArrayLike,
	lam1: float,
	lam2: float,
	P: ArrayLike,
	beta: Optional[float] = None,
	prob_mode: Optional[str] = None,
	clip_prob: bool = True,
	reduce: str = "mean",
) -> Union[float, List[float]]:
	"""
	Compute the hybrid symbolic–neural accuracy functional Ψ(x) over T steps.

	Parameters
	- S, N, alpha, Rcog, Reff, P: Scalars or sequences. Scalars broadcast to match length of the longest input.
	- lam1, lam2: Non-negative regularization weights.
	- beta: Optional probability calibration factor.
	- prob_mode: None | "logit" | "multiply" for calibration style.
	- clip_prob: Whether to clip probabilities to [0, 1].
	- reduce: "mean" | "sum" | "none" aggregation over steps.
	"""
	S_list = _to_list(S)
	N_list = _to_list(N)
	alpha_list = _to_list(alpha)
	Rcog_list = _to_list(Rcog)
	Reff_list = _to_list(Reff)
	P_list = _to_list(P)

	if beta is not None and prob_mode is not None:
		if prob_mode == "logit":
			P_list = calibrate_probability_logit_shift(P_list, beta, clip=clip_prob)
		elif prob_mode == "multiply":
			P_list = calibrate_probability_multiplicative(P_list, beta, clip=clip_prob)
		else:
			raise ValueError("prob_mode must be one of None, 'logit', 'multiply'")

	S_list, N_list, alpha_list, Rcog_list, Reff_list, P_list = _align_shapes(
		S_list, N_list, alpha_list, Rcog_list, Reff_list, P_list
	)

	terms: List[float] = []
	for s, n, a, rc, rf, p in zip(S_list, N_list, alpha_list, Rcog_list, Reff_list, P_list):
		hybrid = (a * s) + ((1.0 - a) * n)
		reg = math.exp(-((lam1 * rc) + (lam2 * rf)))
		prob = _clip(p, 0.0, 1.0) if clip_prob else p
		terms.append(hybrid * reg * prob)

	if reduce == "none":
		return terms
	if reduce == "sum":
		return float(sum(terms))
	if reduce == "mean":
		return float(sum(terms) / len(terms)) if terms else 0.0
	raise ValueError("reduce must be one of 'mean', 'sum', 'none'")


if __name__ == "__main__":
	# Example A: "Single Tracking Step" from the prompt
	S_A = 0.67
	N_A = 0.87
	alpha_A = 0.4
	Rcog_A = 0.17
	Reff_A = 0.11
	lam1_A = 0.6
	lam2_A = 0.4
	P_A = 0.81
	beta_A = 1.2

	psi_A_multiply = compute_psi(
		S_A, N_A, alpha_A, Rcog_A, Reff_A, lam1_A, lam2_A, P_A,
		beta=beta_A, prob_mode="multiply", reduce="mean"
	)
	psi_A_logit = compute_psi(
		S_A, N_A, alpha_A, Rcog_A, Reff_A, lam1_A, lam2_A, P_A,
		beta=beta_A, prob_mode="logit", reduce="mean"
	)

	print(f"Example A (multiply): Ψ(x) = {psi_A_multiply:.6f}")  # expected ~0.667
	print(f"Example A (logit shift): Ψ(x) = {psi_A_logit:.6f}")

	# Example B: "Numerical example from query"
	S_B = [0.65]
	N_B = [0.85]
	alpha_B = [0.3]
	Rcog_B = [0.20]
	Reff_B = [0.15]
	lam1_B = 0.75
	lam2_B = 0.25
	P_corr_B = [0.975]  # already biased and clipped

	psi_B = compute_psi(
		S_B, N_B, alpha_B, Rcog_B, Reff_B, lam1_B, lam2_B, P_corr_B,
		reduce="mean"
	)
	print(f"Example B: Ψ(x) = {psi_B:.12f}")  # expected ~0.63843