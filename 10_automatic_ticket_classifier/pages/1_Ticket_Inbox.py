import streamlit as st
from ticket_utils import get_tickets, update_ticket_status

st.set_page_config(page_title="Ticket Inbox", page_icon="📥", layout="wide")
st.title("📥 Ticket Inbox")
st.caption("Tickets submitted from the main page are stored in the local SQLite database.")

c1, c2 = st.columns(2)
with c1:
    department = st.selectbox("Department", ["All", "HR", "IT", "Transportation"])
with c2:
    status_filter = st.selectbox("Status", ["All", "Open", "In progress", "Resolved"])

tickets = get_tickets(department, status_filter)
if tickets.empty:
    st.info("No tickets match these filters yet. Submit a ticket from the main page to see it here.")
else:
    st.dataframe(tickets[["id", "created_at", "department", "status", "issue_text"]], use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("View or update a ticket")
    ticket_id = st.selectbox("Ticket ID", tickets["id"].tolist())
    ticket = tickets.loc[tickets["id"] == ticket_id].iloc[0]
    st.markdown(f"**Issue**\n\n{ticket['issue_text']}")
    st.markdown(f"**AI policy guidance**\n\n{ticket['ai_answer'] or 'No answer saved.'}")
    new_status = st.selectbox("Update status", ["Open", "In progress", "Resolved"], index=["Open", "In progress", "Resolved"].index(ticket["status"]))
    if st.button("Save status"):
        update_ticket_status(int(ticket_id), new_status)
        st.success("Ticket status updated.")
        st.rerun()
