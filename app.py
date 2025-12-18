import streamlit as st
import sympy as sp
import control
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# FUNÇÕES MATEMÁTICAS
# =========================================================

def asymptotes_info(poles, zeros):
    n = len(poles)
    m = len(zeros)
    q = n - m
    if q <= 0:
        return q, None, []
    sigma = (sum(p.real for p in poles) - sum(z.real for z in zeros)) / q
    angles = [((2*k+1)*180)/q for k in range(q)]
    return q, sigma, angles


def departure_angles(poles, zeros):
    res = []
    for p in poles:
        if abs(p.imag) < 1e-6:
            continue
        ang_zeros = sum(np.degrees(np.angle(p - z)) for z in zeros)
        ang_poles = sum(np.degrees(np.angle(p - pp)) for pp in poles if pp != p)
        theta = (180 + ang_zeros - ang_poles) % 360
        res.append((p, theta))
    return res


def breakaway_points(G_expr):
    s = sp.Symbol("s")
    Kexpr = -1 / G_expr
    dK = sp.diff(Kexpr, s)
    num, _ = sp.fraction(sp.together(dK))
    return sp.nroots(num), Kexpr


def is_on_real_axis_root_locus(x, poles, zeros):
    if abs(x.imag) > 1e-6:
        return False
    xr = x.real
    count = sum(1 for p in poles if abs(p.imag) < 1e-6 and p.real > xr)
    count += sum(1 for z in zeros if abs(z.imag) < 1e-6 and z.real > xr)
    return count % 2 == 1


def eval_K(Kexpr, s_val):
    s = sp.Symbol("s")
    return complex(Kexpr.subs(s, s_val))


def imag_axis_crossing(num, den):
    s, K = sp.Symbol("s"), sp.Symbol("K", real=True)

    n = max(len(num), len(den)) - 1
    num = [0]*(n+1-len(num)) + num
    den = [0]*(n+1-len(den)) + den

    char = sum(den[i]*s**(n-i) for i in range(n+1)) \
         + sum(K*num[i]*s**(n-i) for i in range(n+1))

    coeffs = sp.Poly(char, s).all_coeffs()

    deg = len(coeffs)-1
    rows = deg+1
    cols = int(np.ceil((deg+1)/2))
    R = [[0]*cols for _ in range(rows)]

    R[0][:] = coeffs[0::2] + [0]*(cols-len(coeffs[0::2]))
    R[1][:] = coeffs[1::2] + [0]*(cols-len(coeffs[1::2]))

    for i in range(2, rows):
        for j in range(cols-1):
            R[i][j] = (R[i-1][0]*R[i-2][j+1] - R[i-2][0]*R[i-1][j+1]) / R[i-1][0]

    expr = sp.simplify(R[rows-2][0])
    sols = sp.solve(expr, K)

    results = []
    for sol in sols:
        if sol.is_real and sol > 0:
            sol = float(sol)
            aux = R[rows-3][0]*s**2 + R[rows-3][1]
            roots = sp.solve(aux.subs(K, sol), s)
            omegas = [abs(complex(r).imag) for r in roots if abs(complex(r).real) < 1e-6]
            results.append((sol, omegas))

    return results


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Lugar das Raízes", layout="wide")

SENHA = "lucas123"
senha = st.text_input("Senha", type="password")
if senha != SENHA:
    st.stop()

st.title("Lugar das Raízes — Online")
st.code("Exemplo: 1/((s+1)^3*(s+4))")

expr_str = st.text_input("G(s) =", "1/((s+1)^3*(s+4))")


def parse_tf(expr):
    s = sp.Symbol("s")
    expr = expr.replace("^", "**").replace(")(", ")*(")
    expr = sp.sympify(expr, locals={"s": s})
    num, den = sp.fraction(expr)
    num = [float(c) for c in sp.Poly(num, s).all_coeffs()]
    den = [float(c) for c in sp.Poly(den, s).all_coeffs()]
    return num, den, expr


try:
    num, den, G_expr = parse_tf(expr_str)
    G = control.tf(num, den)

    poles = control.poles(G)
    zeros = control.zeros(G)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Polos")
        st.write(poles)

        st.subheader("Zeros")
        st.write(zeros)

        st.subheader("Assíntotas")
        q, sigma, angs = asymptotes_info(poles, zeros)
        st.write(f"q = {q}")
        if q > 0:
            st.write(f"Centróide: {sigma:.4f}")
            st.write("Ângulos:", angs)

        st.subheader("Ângulos de saída")
        for p, th in departure_angles(poles, zeros):
            st.write(f"{p} → {th:.2f}°")

        st.subheader("Breakaway / Break-in")
        cands, Kexpr = breakaway_points(G_expr)
        for c in cands:
            c = complex(c)
            if abs(c.imag) < 1e-6 and is_on_real_axis_root_locus(c, poles, zeros):
                Kval = eval_K(Kexpr, c.real)
                if abs(Kval.imag) < 1e-6 and Kval.real > 0:
                    st.write(f"s ≈ {c.real:.4f},  K ≈ {Kval.real:.4f}")

        st.subheader("Cruzamento no eixo imaginário")
        crossings = imag_axis_crossing(num, den)
        if crossings:
            for Kc, ws in crossings:
                st.write(f"K crítico = {Kc:.4f}")
                if ws:
                    st.write(f"s = ± j{ws[0]:.4f}")
        else:
            st.write("Nenhum cruzamento encontrado.")

    with col2:
        st.subheader("Lugar das Raízes")
        fig, ax = plt.subplots()
        control.rlocus(G, ax=ax)
        ax.grid(True)
        st.pyplot(fig)

except Exception as e:
    st.error("Erro ao processar G(s)")
    st.code(str(e))
