from helper.libraries import *
from helper.functions import mcq_csv_generate


def mcq_csv_page():
    st.title("Let's Generate Some MCQs from .csv")

    convert_to_mcq = st.file_uploader("Select CSV")
    if convert_to_mcq is not None:
        try:
            convert_to_mcq = pd.read_csv(convert_to_mcq)
            st.dataframe(convert_to_mcq.head())
        except:
            print("File not opened, yet")

        ok = st.button("Generate MCQ")

        if ok:
            generated_csv = mcq_csv_generate(convert_to_mcq)
            st.dataframe(generated_csv.head())

            ct = datetime.datetime.now()
            file = "./generated_csv/" + str(ct) + ".csv"

            generated_csv.to_csv(file)

            with open(file) as f:
                st.download_button('Download CSV', f, str(ct) + '.csv')
                os.remove(file)

            # time.sleep(5)
