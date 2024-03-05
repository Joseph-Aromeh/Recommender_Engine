

# import streamlit as st
# import pickle as pkl

# product = pkl.load(open('Products.pkl', 'rb'))
# similarities = pkl.load(open('user_similarity.pkl', 'rb'))

# product = product['CustomerKey'].values

# st.header('Product Recommender')

# selected_items = st.selectbox('Select a Product from dropdown: ', product)

# def recommend(product):
#     if selected_items not in product.index:
#         return pd.DataFrame(index=product.columns, data={'Product': 0.0, 'WeightedAverage': 0.0})

#     user_history = product.loc[selected_items]
#     weighted_sum = similarities.loc[selected_items] @ product
#     recommendations = weighted_sum[~user_history.astype(bool)]
#     recommendations = recommendations.sort_values(ascending=False)
#     recommendations_df = pd.DataFrame({'Product': recommendations.index, 'Likely_Preference': recommendations.values})

#     return recommendations_df



# if st.button('See Products'):
#     pro = recommend(selected_items)
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(pro[0])
#     with col2:
#         st.text(pro[1])
#     with col3:
#         st.text(pro[2])
#     with col4:
#         st.text(pro[3])
#     with col5:
#         st.text(pro[4])















import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.spatial.distance import cosine


# Load dataset
products = pd.read_pickle('Products.pkl')

# Load similarity matrix
user_similarity = pkl.load(open('user_similarity.pkl', 'rb'))

def recommend_products(customer_key, num_recommendations=5):
    # Find the similarity scores for the given customer key
    similarity_scores = user_similarity[customer_key]

    # Get the indices of products with highest similarity scores
    recommended_indices = np.argsort(similarity_scores)[::-1][:num_recommendations]

    # Get the recommended products
    recommended_products = products.iloc[recommended_indices] 

    return recommended_products['Product']
# Streamlit app
st.title('Product Recommender')

# User input
customer_key = st.number_input('Enter Customer Key:', min_value=0, max_value=products.shape[0]-1, value=0, step=1)

# Display recommendations
recommendations = recommend_products(customer_key)
st.write('Recommended Products:')
st.write(recommendations)

