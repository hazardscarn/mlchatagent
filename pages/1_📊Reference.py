import streamlit as st
if "intermediate_results" not in st.session_state:
    st.session_state.intermediate_results = {}

# Function to display intermediate results
def display_intermediate_results(intermediate_results):
    if intermediate_results:
        st.info("Expand the archive of chat log you want to review")
        st.toast("You can download the intermediate results by howering over table and click download option on top right",
                 icon="🚨",)

        st.markdown("<br>", unsafe_allow_html=True)
        for user_question, results in intermediate_results.items():
            with st.expander(f"Question: {user_question}"):
                for result in results:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Tool Used: {result['tool']}**")
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # st.markdown(f"**SQL Generated:** `{result['sql_generated']}`")
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if result['tool'] == "generate_sql":
                            st.markdown("**Intermediate Steps**")
                            for step in result['intermediate_steps']:
                                st.markdown(f"- **{step['step']}**: {step['details']}")
                        elif result['tool'] == "execute_sql":
                            st.markdown("**Result Data**")
                            st.dataframe(result['result'])
                        elif result['tool'] == "generate_visualizations":
                            st.markdown("**Result Plot**")
                            st.components.v1.html(result['result'], height=350)
                        elif result['tool'] == "subset_shap_explanation":
                            st.markdown("**SHAP Explanation for the subset DataFrame**")
                            st.dataframe(result['result_df'])
                        elif result['tool'] == "Final_Response":
                            st.markdown("**Final Response**")
                            st.markdown(f"**{result['Response']}**")
                    st.markdown("---")
                    
    else:
        st.markdown("No intermediate results to display.")

display_intermediate_results(st.session_state.intermediate_results)
