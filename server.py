from flask import Flask, request, send_file
import os
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

app = Flask(__name__)

MODEL_CACHE_DIR = 'model-cache'

def generateModel(model_name):
  batch_size = 4
  guidance_scale = 15.0
  prompt = model_name

  latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
  )

  return latents

@app.route('/models/<string:filename>')
def download(filename):
  # Extract the name of the model (i.e. "cat" from "cat.txt")
  model_name = os.path.splitext(filename)[0]

  # Check if the file already exists in the cache directory
  cache_filename = os.path.join(MODEL_CACHE_DIR, f"{model_name}.ply")
  if os.path.exists(cache_filename):
    # Send the cached file to the client for download
    return send_file(cache_filename, as_attachment=True)

  latents = generateModel(model_name)

  for i, latent in enumerate(latents):
    with open(cache_filename, 'wb') as f:
      decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)

  # Send the newly-generated file to the client for download
  return send_file(cache_filename, as_attachment=True)

def main():
  # Ensure the model cache directory exists
  os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

  # Start the Flask app
  app.run(debug=True, port=8080, host='0.0.0.0')

if __name__ == '__main__':
  main()