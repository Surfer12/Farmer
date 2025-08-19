import math
from typing import Sequence, Union, List

Number = Union[float, int]
ArrayLike = Union[Number, Sequence[Number]]


def _is_sequence(x: ArrayLike) -> bool:
	return isinstance(x, (list, tuple))


def _to_list(x: ArrayLike) -> List[float]:
	if _is_sequence(x):
		return [float(v) for v in x]  # type: ignore[arg-type]
	return [float(x)]  # type: ignore[arg-type]


def _broadcast_get(values: List[float], idx: int, length: int) -> float:
	if len(values) == 1:
		return values[0]
	if len(values) == length:
		return values[idx]
	raise ValueError("Input lengths must be 1 or match the maximum length.")


def sigmoid(z: Number) -> float:
	return 1.0 / (1.0 + math.exp(-float(z)))


def logit(p: Number, eps: float = 1e-12) -> float:
	p_val = float(p)
	p_clipped = min(max(p_val, eps), 1.0 - eps)
	return math.log(p_clipped / (1.0 - p_clipped))


def apply_beta_bias(p: ArrayLike, beta: float) -> ArrayLike:
	vals = _to_list(p)
	shift = math.log(beta)
	biased = [sigmoid(logit(v) + shift) for v in vals]
	return biased if _is_sequence(p) else biased[0]


def compute_hybrid(S: ArrayLike, N: ArrayLike, alpha: ArrayLike) -> ArrayLike:
	S_list = _to_list(S)
	N_list = _to_list(N)
	A_list = _to_list(alpha)
	length = max(len(S_list), len(N_list), len(A_list))
	out = []
	for i in range(length):
		Si = _broadcast_get(S_list, i, length)
		Ni = _broadcast_get(N_list, i, length)
		Ai = _broadcast_get(A_list, i, length)
		out.append(Ai * Si + (1.0 - Ai) * Ni)
	return out if length > 1 else out[0]


def compute_regularization_factor(Rcog: ArrayLike, Reff: ArrayLike, lam1: float, lam2: float) -> ArrayLike:
	Rc_list = _to_list(Rcog)
	Re_list = _to_list(Reff)
	length = max(len(Rc_list), len(Re_list))
	out = []
	for i in range(length):
		Rci = _broadcast_get(Rc_list, i, length)
		Rei = _broadcast_get(Re_list, i, length)
		out.append(math.exp(-(lam1 * Rci + lam2 * Rei)))
	return out if length > 1 else out[0]


def compute_psi(
	S: ArrayLike,
	N: ArrayLike,
	alpha: ArrayLike,
	Rcog: ArrayLike,
	Reff: ArrayLike,
	lam1: float,
	lam2: float,
	P_corr: ArrayLike,
) -> float:
	"""
	Compute the hybrid symbolic–neural accuracy functional Ψ(x) over discrete steps.

	All array-like inputs may be scalars or sequences. Scalars are broadcast across
	the maximum sequence length. Returns the mean over steps (discrete-time averaging).
	"""
	hybrid = _to_list(compute_hybrid(S, N, alpha))
	reg = _to_list(compute_regularization_factor(Rcog, Reff, lam1, lam2))
	P_list = _to_list(P_corr)
	length = max(len(hybrid), len(reg), len(P_list))
	accum = 0.0
	for i in range(length):
		h = _broadcast_get(hybrid, i, length)
		r = _broadcast_get(reg, i, length)
		p = _broadcast_get(P_list, i, length)
		p = min(max(p, 0.0), 1.0)
		accum += h * r * p
	return accum / float(length)