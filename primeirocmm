import streamlit as st
import sympy as sp
import control
import matplotlib.pyplot as plt

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
    expr_str = expr_str.replace("^", "**")
    expr = sp.sympify(expr_str, locals={"s": s})
    num, den = sp.fraction(expr)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    return num_coeffs, den_coeffs

try:
    num, den = parse_tf(expr_str)
    G = control.tf(num, den)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Polos")
        st.write(control.poles(G))

        st.subheader("Zeros")
        st.write(control.zeros(G))

    with col2:
        st.subheader("Lugar das Raízes")
        fig, ax = plt.subplots()
        control.rlocus(G, ax=ax)
        ax.grid(True)



        st.pyplot(fig)

except Exception as e:
    st.error("Erro ao interpretar a função.")
    st.code(str(e))
