# GameSense

Streamlit app for game recommendations using a PyTorch matrix factorization model.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create a new app.
3. Select the repo, branch, and set **Main file path** to `app.py`.
4. Deploy. Streamlit installs dependencies from `requirements.txt`.

## Required files

The app expects these files in the repository root (or `mf/` for the model):

- `app.py`
- `requirements.txt`
- `best_mf_model.pt` or `mf/best_mf_model.pt`
- `user_to_idx.json`
- `game_to_idx.json`
- `game_names.json`

## Notes

- The model is loaded on CPU for deployment compatibility.
- `runtime.txt` pins the Python version for Streamlit Cloud.
