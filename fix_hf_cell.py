import json
from pathlib import Path
path = Path('hw6/hw6.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
new_source = [
    "# Example: call a Kimi-K3 model via the Hugging Face Inference API",
    "# Set environment variable HF_API_TOKEN with your Hugging Face token before running",
    "import os, requests, json",
    "HF_MODEL = \"audnai/penclaw-Kimi-K3.0-abliterated-GGUF\"  # replace with the exact model id you want",
    "HF_TOKEN = os.environ.get('HF_API_TOKEN')",
    "if not HF_TOKEN:",
    "    raise RuntimeError('Set the HF_API_TOKEN environment variable to call the HF Inference API')",
    "headers = {'Authorization': f'Bearer {HF_TOKEN}'}",
    "# Respect system proxy env vars if present (HTTP_PROXY / HTTPS_PROXY)",
    "proxies = {}",
    "http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')",
    "https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')",
    "if http_proxy:",
    "    proxies['http'] = http_proxy",
    "if https_proxy:",
    "    proxies['https'] = https_proxy",
    "",
    "def hf_infer(model_id, input_text, timeout=120):",
    "    url = f'https://api-inference.huggingface.co/models/{model_id}'",
    "    payload = {'inputs': input_text, 'options': {'wait_for_model': True}}",
    "    try:",
    "        if proxies:",
    "            r = requests.post(url, headers=headers, json=payload, timeout=timeout, proxies=proxies)",
    "        else:",
    "            r = requests.post(url, headers=headers, json=payload, timeout=timeout)",
    "        r.raise_for_status()",
    "    except requests.exceptions.RequestException as e:",
    "        raise RuntimeError(\n",
    "            f'Network error calling Hugging Face Inference API: {e}. '\n",
    "            'Check DNS, proxy, VPN, or set HTTP(S)_PROXY environment variables if required.'\n",
    "        ) from e",
    "    out = r.json()",
    "    if isinstance(out, dict) and out.get('error'):",
    "        raise RuntimeError(out['error'])",
    "    if isinstance(out, list):",
    "        first = out[0]",
    "        return first.get('generated_text', str(first))",
    "    return str(out)",
    "",
    "# Run the HF Kimi-K3 model on one sample patient (uses proxies if set)",
    "example_patient = sample_patients.iloc[0]",
    "example_text = serialize_patient(example_patient)",
    "print('Calling HF Kimi-K3 model...')",
    "try:",
    "    response_text = hf_infer(HF_MODEL, example_text)",
    "    print('Kimi-K3 response:')",
    "    print(response_text)",
    "except RuntimeError as e:",
    "    print('Error while calling Hugging Face Inference API:')",
    "    print(e)",
    "    print('If this is a network/DNS error, consider running the model locally via GGUF and llama.cpp (see next cell).')",
]
updated = False
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and any('HF_MODEL = "audnai/penclaw' in line for line in cell.get('source', [])):
        cell['source'] = new_source
        updated = True
        break
if not updated:
    raise FileNotFoundError('Could not find the HF inference cell to patch')
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Patched HF inference cell successfully')
