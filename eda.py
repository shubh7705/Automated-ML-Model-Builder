import pandas as pd
import streamlit as st
import ydata_profiling
from ydata_profiling import ProfileReport

def data_info(data):
    prof = ProfileReport(data)
    prof.to_file(output_file='output.html')

    with open("output.html", "r", encoding="utf-8") as f:
       html_content = f.read()

    # Display the HTML in Streamlit
    st.components.v1.html(html_content, height=800, scrolling=True)
