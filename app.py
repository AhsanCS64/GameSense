import streamlit as st
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Page layout and branding
st.set_page_config(page_title="GameSense Recommender", page_icon="🎮", layout="centered")

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
    base_dir = Path(__file__).resolve().parent

    # Load configuration indices
    user_to_idx_path = base_dir / 'user_to_idx.json'
    game_to_idx_path = base_dir / 'game_to_idx.json'
    game_names_path = base_dir / 'game_names.json'

    with open(user_to_idx_path, 'r', encoding='utf-8') as f:
        user_to_idx = json.load(f)
    with open(game_to_idx_path, 'r', encoding='utf-8') as f:
        game_to_idx = json.load(f)

    # Load or generate game name mappings
    if game_names_path.exists():
        with open(game_names_path, 'r', encoding='utf-8') as f:
            id_to_name = json.load(f)
    else:
        id_to_name = {g: g for g in game_to_idx.keys()}

    # Ensure every game in game_to_idx has a readable name
    for g_id in game_to_idx.keys():
        if g_id not in id_to_name:
            id_to_name[g_id] = g_id

    name_to_id = {v: k for k, v in id_to_name.items()}
    idx_to_game_id = {int(v): k for k, v in game_to_idx.items()}

    # Resolve model path (supports mf/best_mf_model.pt or best_mf_model.pt)
    model_path = base_dir / 'mf' / 'best_mf_model.pt'
    if not model_path.exists():
        model_path = base_dir / 'best_mf_model.pt'

    # Initialize model
    model = MatrixFactorization(len(user_to_idx), len(game_to_idx), embedding_dim=16)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, user_to_idx, game_to_idx, idx_to_game_id, id_to_name, name_to_id

try:
    model, user_to_idx, game_to_idx, idx_to_game_id, id_to_name, name_to_id = load_recommender()

    st.title("🎮 GameSense Recommender")
    st.markdown("Discover your next favorite game using collaborative filtering and neural embeddings.")

    # Mode selection: By Game Name or By User
    mode = st.radio(
        "Recommendation Mode:",
        ["🎯 Recommend by Game Name (Find Similar Games)", "👤 Recommend for User ID"],
        horizontal=True
    )

    if mode.startswith("🎯"):
        st.subheader("Find Similar Games")
        # List sorted game names for selection
        game_names_list = sorted(list(name_to_id.keys()))
        selected_game_name = st.selectbox("Select a Game Name:", game_names_list)

        if selected_game_name:
            selected_game_id = name_to_id[selected_game_name]
            target_idx = game_to_idx[selected_game_id]

            # Compute similarity using learned item embeddings
            with torch.no_grad():
                item_embeddings = model.item_emb.weight.data
                target_emb = item_embeddings[target_idx].unsqueeze(0)
                sims = F.cosine_similarity(target_emb, item_embeddings).cpu().numpy()

            # Rank items by cosine similarity (excluding the selected game itself)
            sims[target_idx] = -float('inf')
            top_indices = np.argsort(-sims)[:10]

            st.write(f"### Top 10 Games similar to **{selected_game_name}**:")
            for i, idx in enumerate(top_indices, 1):
                g_id = idx_to_game_id[idx]
                g_name = id_to_name.get(g_id, g_id)
                match_pct = max(0.0, float(sims[idx])) * 100
                st.write(f"{i}. **{g_name}** `({match_pct:.1f}% match) - {g_id}`")

    else:
        st.subheader("Personalized User Recommendations")
        user_options = sorted(list(user_to_idx.keys()), key=lambda x: int(x) if x.isdigit() else x)
        selected_user = st.selectbox("Select a User ID:", user_options)

        if selected_user:
            u_idx = user_to_idx[selected_user]

            all_game_ids = list(game_to_idx.keys())
            candidate_game_idxs = torch.tensor([game_to_idx[g] for g in all_game_ids], dtype=torch.long)
            u_tensor = torch.tensor([u_idx] * len(all_game_ids), dtype=torch.long)

            with torch.no_grad():
                logits = model(u_tensor, candidate_game_idxs)
                scores = torch.sigmoid(logits).cpu().numpy()

            ranked_indices = np.argsort(-scores)
            top_game_ids = [all_game_ids[idx] for idx in ranked_indices[:10]]

            st.write(f"### Top 10 Game Recommendations for User **{selected_user}**:")
            for i, g_id in enumerate(top_game_ids, 1):
                g_name = id_to_name.get(g_id, g_id)
                st.write(f"{i}. **{g_name}** `({g_id})`")

except Exception as e:
    st.error(f"Error initializing recommender system: {e}")
    st.info("Please ensure 'best_mf_model.pt', 'user_to_idx.json', 'game_to_idx.json', and 'game_names.json' are in the project folder.")
