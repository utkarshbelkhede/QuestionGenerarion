from helper.libraries import *
from helper.mcq_sentence_page import mcq_sentence_page
from helper.mcq_csv_page import mcq_csv_page


def side_menu():
    choice = st.sidebar.selectbox("Generate", ("MCQs", "FIBs"))

    if choice == "MCQs":
        page = st.sidebar.selectbox("How?", ("from sentence", "from csv"))
        if page == "from sentence":
            mcq_sentence_page()
        elif page == "from csv":
            mcq_csv_page()
    elif choice == "FIBs":
        pass


if __name__ == '__main__':
    side_menu()
