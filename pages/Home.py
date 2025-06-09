import streamlit as st

if not st.session_state.get("logged_in", False):
    st.warning("You must log in to access this page.")
    st.rerun()

st.set_page_config(page_title="Options App", layout="wide")
st.title("Welcome to the Options Payoff App")

st.write("Use the sidebar to navigate between the Strategy Builder and Greek Calculator.")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.switch_page("main.py")