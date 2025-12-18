import streamlit as st
import sympy as sp
import control
import matplotlib.pyplot as plt

import numpy as np
import sympy as sp

def asymptotes_info(poles, zeros):
    poles = list(poles)
    zeros = list(zeros)
    n = len(poles)
    m = len(zeros)
    q = n - m
    if q <= 0:
        return q, None, []
    sigma = (sum([p.real for p in poles]) - sum([z.real for z in zeros])) / q
    angles = [((2*k+1)*180.0)/q for k in range(q)]
    return q, sigma, angles

def departure_angles(poles, zeros):
    # Só faz sentido para polos complexos (imaginário != 0)
    # Fórmula: θd = 180° + Σ∠(p - zeros) - Σ∠(p - outros polos)
    res = []
    for p in poles:
        if abs(p.imag) < 1e-9:
            continue
        ang_zeros = 0.0
        for z in zeros:
            ang_zeros += np.degrees(np.angle(p - z))
        ang_poles = 0.0
        for pp in poles:
            if pp == p:
                continue
            ang_poles += np.degrees(np.angle(p - pp))
        theta = 180.0 + ang_zeros - ang_poles
        # normaliza 0..360
        theta = (theta % 360.0 + 360.0) % 360.0
        res.append((p, theta))
    return res

def k_of_s(G_expr, s_symbol):
    # G_expr é expressão sympy de G(s) (sem K)
    # K(s) = -1/G(s) ao impor 1 + K G(s) = 0
    return sp.simplify(-1 / G_expr)

def breakaway_points(G_expr):
    # resolve dK/ds = 0
    s = sp.Symbol('s')
    Kexpr = k_of_s(G_expr, s)
    dK = sp.diff(Kexpr, s)
    # Coloca como fração e pega numerador
    num, den = sp.fraction(sp.together(dK))
    # Raízes do numerador = candidatos
    sols = sp.nroots(num)  # numérico (robusto)
    return sols, Kexpr

def is_on_real_axis_root_locus(x, poles, zeros, tol=1e-6):
    # Regra do eixo real: nº de polos+zeros à direita do ponto é ímpar
    if abs(x.imag) > tol:
        return False
    xr = float(x.real)
    count = 0
    for p in poles:
        if abs(p.imag) < tol and p.real > xr:
            count += 1
    for z in zeros:
        if abs(z.imag) < tol and z.real > xr:
            count += 1
    return (count % 2) == 1

def eval_K_at_point(Kexpr, point):
    s = sp.Symbol('s')
    val = complex(Kexpr.subs(s, point))
    return val

def routh_table(coeffs):
    # coeffs: lista do polinômio característico em ordem decrescente
    # Retorna tabela (lista de listas) e primeira coluna
    coeffs = [float(c) for c in coeffs]
    n = len(coeffs) - 1
    rows = n + 1
    cols = int(np.ceil((n + 1) / 2))

    R = [[0.0]*cols for _ in range(rows)]
    # preenche 1ª e 2ª linha
    R[0][:] = coeffs[0::2] + [0.0]*(cols - len(coeffs[0::2]))
    R[1][:] = coeffs[1::2] + [0.0]*(cols - len(coeffs[1::2]))

    eps = 1e-9
    for i in range(2, rows):
        for j in range(cols-1):
            a = R[i-1][0] if abs(R[i-1][0]) > eps else eps
            R[i][j] = (a*R[i-2][j+1] - R[i-2][0]*R[i-1][j+1]) / a

        # Caso de linha toda zero (auxiliary polynomial) — implementação simples
        if all(abs(v) < eps for v in R[i]):
            # monta polinômio auxiliar da linha acima (i-1)
            order = n - (i-1)
            aux = []
            for k in range(cols):
                power = order - 2*k
                if power < 0:
                    break
                aux.append(R[i-1][k])
            # derivada do auxiliar preenche a linha zero
            # (versão numérica aproximada)
            # Aqui, uma saída simples: colocar eps na primeira coluna pra seguir
            R[i][0] = eps

    first_col = [R[i][0] for i in range(rows)]
    return R, first_col


# ---------------- LOGIN SIMPLES ----------------
st.set_page_config(page_title="Root Locus", layout="wide")

SENHA_CORRETA = "lucas123"  # MUDE DEPOIS

senha = st.text_input("Senha", type="password")
if senha != SENHA_CORRETA:
    st.stop()

# ---------------- APP ----------------
st.title("Lugar das Raízes — Online")

st.write("Digite G(s). Ex:")
st.code("(s^2+5*s+6)/(s^5+17*s^4+106*s^3+298*s^2+388*s+240)")

expr_str = st.text_input("G(s) =", "(s^2+5*s+6)/(s^5+17*s^4+106*s^3+298*s^2+388*s+240)")

def parse_tf(expr_str):
    s = sp.Symbol('s')
    expr_str = expr_str.replace("^", "**").replace(")(", ")*(")
    expr = sp.sympify(expr_str, locals={"s": s})
    num, den = sp.fraction(sp.together(expr))

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    return num_coeffs, den_coeffs, expr


try:
    num, den, G_expr = parse_tf(expr_str)
    G = control.tf(num, den)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Polos")
        st.write(control.poles(G))

        st.subheader("Zeros")
        st.write(control.zeros(G))
st.subheader("3.1 Assíntotas")
q, sigma, angles = asymptotes_info(poles, zeros)
st.write(f"q = #polos - #zeros = {len(poles)} - {len(zeros)} = **{q}**")
if q > 0:
    st.write(f"Centróide (σa): **{sigma:.4f}**")
    st.write("Ângulos:", [f"{a:.2f}°" for a in angles])
else:
    st.write("Não há assíntotas (n ≤ m).")

st.subheader("3.2 Ângulos de saída (polos complexos)")
deps = departure_angles(poles, zeros)
if deps:
    for p, th in deps:
        st.write(f"p = {p:.4g}  →  θd = **{th:.4f}°**")
else:
    st.write("Nenhum polo complexo para calcular ângulo de saída.")

st.subheader("3.3 Breakaway / Break-in (candidatos)")
cands, Kexpr = breakaway_points(G_expr)
shown = 0
for c in cands:
    c = complex(c)
    # normalmente breakaway útil é no eixo real
    if abs(c.imag) < 1e-6 and is_on_real_axis_root_locus(c, poles, zeros):
        Kval = eval_K_at_point(Kexpr, c.real)
        # geralmente consideramos K real positivo
        if abs(Kval.imag) < 1e-3 and Kval.real > 0:
            st.write(f"s ≈ {c.real:.4f}  |  K ≈ {Kval.real:.4f}")
            shown += 1
if shown == 0:
    st.write("Nenhum candidato real válido (K>0 e no trecho do LR) encontrado. Mostrando candidatos brutos:")
    st.write([complex(c) for c in cands])

st.subheader("3.4 Routh-Hurwitz (estabilidade em função de K)")
st.write("Aqui dá para implementar a tabela simbólica em K (mais completo).")
st.write("Como MVP, dá pra começar com um teste numérico: varrer K e verificar se algum polo cruza para Re>0.")

                               

    with col2:
        st.subheader("Lugar das Raízes")
        fig, ax = plt.subplots()
        control.rlocus(G, ax=ax)
        ax.grid(True)



        st.pyplot(fig)

except Exception as e:
    st.error("Erro ao interpretar a função.")
    st.code(str(e))
