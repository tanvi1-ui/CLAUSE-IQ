import streamlit as st

from rag_pipeline import get_chain


st.set_page_config(

    page_title="ClauseIQ",

    layout="wide"
)

st.title(

    "ClauseIQ"
)

st.write(

    "AI-Powered Insurance Policy Reasoning System"
)

query = st.text_input(

    "Enter your insurance query"
)

if st.button(

    "Evaluate"
):

    chain = get_chain()

    result = chain.invoke(

        {

            "query": query
        }
    )

    st.subheader(

        "Decision"
    )

    st.write(

        result["result"]
    )

    st.subheader(

        "Relevant Clauses"
    )

    for i, doc in enumerate(

        result["source_documents"],

        start=1
    ):

        st.write(

            f"Clause {i}"
        )

        st.write(

            doc.page_content
        )

        st.divider()