import argparse
from .core import compute_psi, apply_beta_bias


def example_runtime(bias_mode: str):
	S = 0.65
	N = 0.85
	alpha = 0.3
	Rcog = 0.20
	Reff = 0.15
	lam1 = 0.75
	lam2 = 0.25
	if bias_mode == "pre":
		P_base = 0.81
		beta = 1.2
		P_corr = apply_beta_bias(P_base, beta)
	else:
		# Post-bias P provided by the example (~0.975)
		P_corr = 0.975
	psi = compute_psi(S, N, alpha, Rcog, Reff, lam1, lam2, P_corr)
	return psi


def example_open_source(bias_mode: str):
	S = 0.74
	N = 0.84
	alpha = 0.5
	Rcog = 0.14
	Reff = 0.09
	lam1 = 0.55
	lam2 = 0.45
	if bias_mode == "pre":
		P_base = 0.77
		beta = 1.3
		P_corr = apply_beta_bias(P_base, beta)
	else:
		# Post-bias P is capped to 1.0 in the example
		P_corr = 1.0
	psi = compute_psi(S, N, alpha, Rcog, Reff, lam1, lam2, P_corr)
	return psi


def main():
	parser = argparse.ArgumentParser(description="Hybrid functional examples")
	parser.add_argument("--which", choices=["runtime", "open_source"], default="runtime")
	parser.add_argument("--bias-mode", choices=["pre", "post"], default="post", help="pre: apply beta; post: use provided post-bias P")
	args = parser.parse_args()

	if args.which == "runtime":
		psi = example_runtime(args.bias_mode)
		print(f"Ψ(x) [runtime example, {args.bias_mode}-bias]: {psi}")
	else:
		psi = example_open_source(args.bias_mode)
		print(f"Ψ(x) [open-source example, {args.bias_mode}-bias]: {psi}")


if __name__ == "__main__":
	main()