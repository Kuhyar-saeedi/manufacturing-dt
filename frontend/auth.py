"""Streamlit auth helpers — JWT session state, page guards, sidebar user widget."""

import os
import streamlit as st
import requests

PLANT_NAMES = {
    "alpha": "Plant Alpha",
    "beta":  "Plant Beta",
    "gamma": "Plant Gamma",
}


def _api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")


def _do_login(username: str, password: str) -> bool:
    try:
        base = _api_base()
        resp = requests.post(
            f"{base}/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code != 200:
            return False
        token = resp.json()["access_token"]
        me = requests.get(
            f"{base}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        ).json()
        st.session_state["auth_token"]    = token
        st.session_state["auth_user"]     = me["username"]
        st.session_state["auth_role"]     = me["role"]
        st.session_state["auth_plant_id"] = me.get("plant_id")
        return True
    except Exception:
        return False


def _show_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if _do_login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.info("Demo: `admin / admin` (all plants)  |  `operator / operator` (Plant Alpha)")


def require_auth():
    """Show login form and stop the page if not authenticated."""
    if "auth_token" not in st.session_state:
        _show_login()
        st.stop()


def require_role(role: str):
    """Require a specific role; stops with an error if not met."""
    require_auth()
    if st.session_state.get("auth_role") != role:
        st.error(f"Access denied — this page requires the **{role}** role.")
        if st.button("Back to Dashboard"):
            st.switch_page("app.py")
        st.stop()


def get_auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}


def get_current_user() -> dict:
    return {
        "username": st.session_state.get("auth_user", ""),
        "role":     st.session_state.get("auth_role", ""),
        "plant_id": st.session_state.get("auth_plant_id"),
    }


def get_available_plants() -> list:
    """Return plant IDs accessible to the current user."""
    if st.session_state.get("auth_role") == "admin":
        return list(PLANT_NAMES.keys())
    assigned = st.session_state.get("auth_plant_id", "alpha")
    return [assigned] if assigned else ["alpha"]


def render_sidebar_user() -> str:
    """Render user info, plant selector, and logout in the sidebar. Returns selected plant_id."""
    user = get_current_user()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**User:** {user['username']}  \n"
        f"**Role:** {user['role'].capitalize()}"
    )

    available = get_available_plants()
    plant_labels = [PLANT_NAMES.get(p, p.capitalize()) for p in available]

    if len(available) == 1:
        st.sidebar.markdown(f"**Plant:** {plant_labels[0]}")
        selected_plant = available[0]
    else:
        chosen_label = st.sidebar.selectbox("Plant", plant_labels, key="plant_selector")
        selected_plant = available[plant_labels.index(chosen_label)]

    if st.sidebar.button("Logout", use_container_width=True):
        for k in ["auth_token", "auth_user", "auth_role", "auth_plant_id"]:
            st.session_state.pop(k, None)
        st.rerun()

    return selected_plant
