"""
Simple Streamlit test to verify the setup
"""

import streamlit as st
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(
    page_title="SpiralDelta-DB Test",
    page_icon="🌀",
    layout="wide"
)

st.title("🌀 SpiralDelta-DB Test Dashboard")
st.write("This is a simple test to verify Streamlit is working properly.")

# Test basic functionality
if st.button("Test Basic Functionality"):
    st.success("✅ Streamlit is working!")
    
    # Test numpy
    test_array = np.random.rand(5, 3)
    st.write("**Random numpy array:**")
    st.write(test_array)
    
    # Test pandas
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    st.write("**Test pandas DataFrame:**")
    st.dataframe(test_df)

st.write("If you can see this message, the basic setup is working correctly!")