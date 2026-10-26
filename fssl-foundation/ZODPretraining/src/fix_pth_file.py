import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
saved_models = os.path.join(script_dir, "saved_models")

input_file = os.path.join(saved_models, "centralized_ResNet_checkpoint_DINO_on_ZOD.pth")
checkpoint = torch.load(input_file, map_location="cpu")

# Se il file di FB ha i pesi dentro una sotto-chiave "model", estraiamola
if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# CREIAMO IL NUOVO DIZIONARIO CON IL PREFISSO 'backbone.'
fixed_state_dict = {}
for k, v in state_dict.items():
    # Saltiamo il layer 'fc' (la testa di ImageNet) perché non ci serve
    if k.startswith("fc."):
        continue
    # AGGIUNGIAMO IL PREFISSO MAGICO
    fixed_state_dict[f"backbone.{k}"] = v

# Salviamo il nuovo file
output_file = os.path.join(saved_models, "centralized_ResNet_checkpoint_DINO_on_ZOD_FIXED.pth")
torch.save(fixed_state_dict, output_file)

print(f"Conversione completata! Creato: {output_file}")
print(f"Numero chiavi convertite: {len(fixed_state_dict)}")