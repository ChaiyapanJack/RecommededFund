import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the machine learning model and fund_tax_list
with open('model_recoment_fund.pkl', 'rb') as file:
    model, fund_tax_list = pickle.load(file)

# Function to recommend funds
def recommend_items(input_item, model, items_to_recommend):
    possible_items = fund_tax_list
    if input_item in possible_items :
        predictions = [(input_item, item, model.predict(input_item, item).est) for item in possible_items if item not in input_item]
        recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:items_to_recommend]
        recommended_items = [item for _, item, _ in recommendations]
    else :
        recommended_items = ['ไม่ใช่กองทุนภาษี']
        
    return recommended_items

# Streamlit UI
st.title('Fund Tex Recommendation App')

# Create two tabs
tab1, tab2 = st.tabs( ["📈 Recomended Fund", " 🗃 Recommend from CSV"])

with tab1:
    st.header("📈 : Predict Recommended Fund")

    # Input field for fund recommendation
    input_item = st.text_input("Enter a Fund Name:", "")

    # Button to recommend funds
    if st.button("Recommend Funds"):
        recommended_items = recommend_items(input_item, model, 10)
        st.write(f"Recommended items based on {input_item}: {recommended_items}")

with tab2:
    st.header("🗃 : Recommend from CSV")
    st.caption('*** File detail with [Fund Name]')

    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Predict and display results
        recommendations = [recommend_items(input_item, model, 10) for input_item in df['Fund Name']]
        df['Recommended Funds'] = recommendations
        st.write('Predicted DataFrame:')
        st.write(df)

        image = Image.open('top_fund.png')

        st.image(image, caption='Sunrise by the mountains')
        # Create a button to download the predicted DataFrame as a CSV file
        st.download_button("Download CSV", df.to_csv(index=False), file_name="recommended_funds.csv")

