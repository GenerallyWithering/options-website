import streamlit as st
from login import login_page
from database import init_db

# Initialize DB if it doesn't exist
init_db()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    st.switch_page("pages/Home.py")