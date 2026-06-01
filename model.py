import pickle, json
from pathlib import Path

# Load the pickled model
with open("models/registry/champion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Save using XGBoost's native format
model.save_model("models/registry/champion_model.ubj")
print("Saved as champion_model.ubj")
