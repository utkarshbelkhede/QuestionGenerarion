from helper.libraries import *
from helper.functions import getMCQs, mcq_sent


def mcq_sentence_page():
    st.title("Let's Generate Some MCQ from a sentence")

    sentence = st.text_area('Sentence')
    target = st.text_input('Target')

    if target.lower() not in sentence.lower():
        st.write("Target is not in Sentence.")
    else:
        ok = st.button("Generate MCQ")

        if ok:
            question, answer, distractors, meaning = getMCQs(mcq_sent(sentence.lower(), target.lower()))
            st.write("**Question:**", question)

            if len(distractors) == 1:
                st.write(distractors[0])
            else:
                mcq = distractors[:3]
                mcq.append(answer)
                random.shuffle(mcq)
                st.write("**Options:**")
                for i in range(len(mcq)):
                    st.write(i + 1, mcq[i])
                st.write("**Answer:**", answer)
                st.write("**Meaning:**", meaning)
