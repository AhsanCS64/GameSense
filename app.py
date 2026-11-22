
import streamlit as st
import json
import numpy as np
import torch
import torch.nn as nn

# Define the MatrixFactorization Model architecture
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_indices, item_indices):
        u_vecs = self.user_emb(user_indices)
        i_vecs = self.item_emb(item_indices)
        dots = (u_vecs * i_vecs).sum(dim=1)
        u_b = self.user_bias(user_indices).squeeze()
        i_b = self.item_bias(item_indices).squeeze()
        logits = self.global_bias + u_b + i_b + dots
        return logits

@st.cache_resource
def load_recommender():
    # Load configuration indices
    with open('user_to_idx.json', 'r') as f:
        user_to_idx = json.load(f)
    with open('game_to_idx.json', 'r') as f:
        game_to_idx = json.load(f)
    
    idx_to_game = {int(v): k for k, v in game_to_idx.items()}

    # Initialize model
    model = MatrixFactorization(len(user_to_idx), len(game_to_idx), embedding_dim=16)
    model.load_state_dict(torch.load('best_mf_model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    return model, user_to_idx, game_to_idx, idx_to_game

try:
    model, user_to_idx, game_to_idx, idx_to_game = load_recommender()
    st.title("🎮 GameSense Live Recommender")
    st.markdown("Suggesting the top games you haven't played using learned collaborative filtering neural embeddings.")

    # Select user ID
    user_options = sorted(list(user_to_idx.keys()))
    selected_user = st.selectbox("Select a User ID to recommend for:", user_options)

    if selected_user:
        u_idx = user_to_idx[selected_user]
        
        # Prepare all game candidates
        all_games = list(game_to_idx.keys())
        candidate_game_idxs = torch.tensor([game_to_idx[g] for g in all_games], dtype=torch.long)
        u_tensor = torch.tensor([u_idx] * len(all_games), dtype=torch.long)

        with torch.no_grad():
            logits = model(u_tensor, candidate_game_idxs)
            scores = torch.sigmoid(logits).cpu().numpy()

        # Sort game recommendations descending
        ranked_indices = np.argsort(-scores)
        recommended_games = [all_games[idx] for idx in ranked_indices[:10]]

        st.subheader(f"Top 10 Game Recommendations for {selected_user}")
        for i, game in enumerate(recommended_games, 1):
            st.write(f"{i}. **{game}**")

except Exception as e:
    st.error(f"Error initializing recommender system: {e}")
    st.info("Please ensure 'best_mf_model.pt', 'user_to_idx.json', and 'game_to_idx.json' are in the same folder as this app.")
