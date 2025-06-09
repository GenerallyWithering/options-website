import streamlit as st
from database import SessionLocal, User, LoginLog
from auth import get_password_hash, authenticate_user

def login_page():
    if 'login_failed' not in st.session_state:
        st.session_state.login_failed = False
    if 'show_recover' not in st.session_state:
        st.session_state.show_recover = False

    if st.session_state.show_recover:
        recover_password()
        if st.button("Back to Login/Register"):
            st.session_state.show_recover = False
            st.session_state.login_failed = False
            st.rerun()
    else:
        tab = st.tabs(["Login", "Register"])

        with tab[0]:
            login()
            # Show "Forgot Password?" only if login failed
            if st.session_state.login_failed:
                if st.button("Forgot Password?"):
                    st.session_state.show_recover = True
                    st.rerun()

        with tab[1]:
            register()


def login():
    st.title("Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Login")

    if login_button:
        db = SessionLocal()
        # Log the login attempt with timestamp
        log = LoginLog(username=username)
        db.add(log)
        db.commit()

        user = authenticate_user(db, username, password)
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state.login_failed = False
            st.session_state.show_recover = False
            db.close()
            st.success(f"Logged in as {username}")
            st.rerun()  # refresh to trigger navigation outside this file
        else:
            st.session_state.login_failed = True
            st.error("Invalid username or password")
        db.close()


def register():
    st.title("Register")

    username = st.text_input("Choose a username", key="reg_username")
    email = st.text_input("Enter your email", key="reg_email")
    password = st.text_input("Choose a password", type="password", key="reg_password")
    password2 = st.text_input("Confirm password", type="password", key="reg_password2")
    password_hint = st.text_input("Set a password hint (optional)", key="reg_hint")
    register_button = st.button("Register")

    if register_button:
        if password != password2:
            st.error("Passwords do not match")
            return

        db = SessionLocal()
        if db.query(User).filter(User.username == username).first():
            st.error("Username already taken")
            db.close()
            return

        if db.query(User).filter(User.email == email).first():
            st.error("Email already registered")
            db.close()
            return

        new_user = User(
            username=username,
            email=email,
            hashed_password=get_password_hash(password),
            password_hint=password_hint if password_hint else None
        )
        db.add(new_user)
        db.commit()
        db.close()
        st.success("Registration successful! You can now log in.")


def recover_password():
    st.title("Recover Password")

    username_or_email = st.text_input("Enter your username or email", key="recover_username_or_email")
    recover_button = st.button("Show Password Hint")

    if recover_button:
        db = SessionLocal()
        user = db.query(User).filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        db.close()

        if user:
            if user.password_hint:
                st.info(f"Password hint: {user.password_hint}")
            else:
                st.info("No password hint was set for this account.")
        else:
            st.error("User not found.")


def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.rerun()
