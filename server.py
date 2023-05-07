from flask import Flask, request, send_file
import os
import torch
import trimesh
import math

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
PI = 3.14159265359

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
  is_ply = filename.endswith('.ply')
  is_glb = filename.endswith('.glb')
  # return 400 if the filename does not end with ".ply" or ".glb"
  if not is_ply and not is_glb:
    return 'Invalid filename', 400

  # Extract the name of the model (i.e. "cat" from "cat.ply")
  model_name = os.path.splitext(filename)[0]

  print(f"Query for model: {model_name}")

  # Check if the file already exists in the cache directory
  ply_filename = os.path.join(MODEL_CACHE_DIR, f"{model_name}.ply")
  glb_filename = os.path.join(MODEL_CACHE_DIR, f"{model_name}.glb")
  if is_ply and os.path.exists(ply_filename):
    print(f"Found cached model: {ply_filename}")
    # Send the cached file to the client for download
    return send_file(ply_filename, as_attachment=True)

  if is_glb and os.path.exists(glb_filename):
    print(f"Found cached model: {glb_filename}")
    # Send the cached file to the client for download
    return send_file(glb_filename, as_attachment=True)

  # Generate the model
  print(f"Generating model: {model_name}")
  latents = generateModel(model_name)

  for i, latent in enumerate(latents):
    with open(ply_filename, 'wb') as f:
      decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)

    mesh = trimesh.load(f.name)
    rot = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    rot = trimesh.transformations.rotation_matrix(math.pi, [0, 1, 0])
    mesh = mesh.apply_transform(rot)

    mesh.export(glb_filename, file_type='glb')

  # Send the newly-generated file to the client for download
  return send_file(ply_filename if is_ply else glb_filename, as_attachment=True)

def main():
  # Ensure the model cache directory exists
  os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

  # Start the Flask app
  app.run(debug=True, port=8081, host='0.0.0.0')

if __name__ == '__main__':
  main()