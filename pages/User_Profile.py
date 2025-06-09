import streamlit as st
from database import SessionLocal, User
from auth import get_password_hash, verify_password

if not st.session_state.get("logged_in", False):
    st.warning("You must log in to access this page.")
    st.rerun()


if not st.session_state.get("logged_in", False):
    st.warning("You must log in to access this page.")
    st.stop()

st.title("User Profile")

username = st.session_state.get("username")

db = SessionLocal()
user = db.query(User).filter(User.username == username).first()

if not user:
    st.error("User data not found.")
    st.stop()

with st.form("profile_form"):
    st.write(f"**Username:** {user.username} (cannot change)")

    email = st.text_input("Email", value=user.email)
    password_hint = st.text_input("Password Hint", value=user.password_hint if user.password_hint else "")

    st.write("### Change Password")
    current_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_new_password = st.text_input("Confirm New Password", type="password")

    submit = st.form_submit_button("Save Changes")

if submit:
    errors = []

    # Validate email is not empty
    if not email.strip():
        errors.append("Email cannot be empty.")

    # Check if email is changed and unique
    if email != user.email:
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            errors.append("This email is already registered by another user.")

    # Validate password change only if new password is provided
    if new_password or confirm_new_password:
        if not current_password:
            errors.append("You must enter your current password to change password.")
        elif not verify_password(current_password, user.hashed_password):
            errors.append("Current password is incorrect.")
        elif new_password != confirm_new_password:
            errors.append("New passwords do not match.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Update email and password hint
        user.email = email
        user.password_hint = password_hint if password_hint.strip() else None

        # Update password if changed
        if new_password:
            user.hashed_password = get_password_hash(new_password)

        db.commit()
        st.success("Profile updated successfully.")

db.close()

if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.switch_page("pages/Home.py")
    st.rerun()
    
