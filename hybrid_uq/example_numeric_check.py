def main():
    S = 0.78
    N = 0.86
    alpha = 0.48
    O = alpha * S + (1 - alpha) * N  # 0.8216

    R_cog = 0.13
    R_eff = 0.09
    lambda1 = 0.57
    lambda2 = 0.43
    total = lambda1 * R_cog + lambda2 * R_eff  # 0.1128

    import math
    pen = math.exp(-total)  # ~0.893

    P = 0.80
    beta = 1.15
    post = min(beta * P, 1.0)  # 0.92

    psi = O * pen * post

    print({
        "O": O,
        "total": total,
        "pen": pen,
        "post": post,
        "Psi": psi,
    })


if __name__ == "__main__":
    main()