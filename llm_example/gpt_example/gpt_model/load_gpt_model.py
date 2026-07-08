import torch
import tiktoken
from llm_example.gpt_example.config import GPT_CONFIG_124M
from model.gpt_model import GPTModel
from tools.generate_text import generate, text_to_token_ids, token_ids_to_text
from data.tools.gpt_download import load_weights_from_ckpt
from llm_example.gpt_example.gpt_model.load_weights_into_gpt import load_weights_into_gpt
from tools.log import log

def load_gpt_model(device=torch.device("cpu"),model_name="gpt2-small (124M)",context_length=1024,qkv_bias=True):
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": context_length})
    NEW_CONFIG.update({"qkv_bias": qkv_bias})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = load_weights_from_ckpt(model_size=model_size, models_dir="data/output/gpt2")
    load_weights_into_gpt(gpt, params)

    gpt.to(device)

    return gpt, NEW_CONFIG

def main():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt, config= load_gpt_model(device=device)

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=config["context_length"],
        top_k=50,
        temperature=1.5
    )
    log.info("Output text:\n%s", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
