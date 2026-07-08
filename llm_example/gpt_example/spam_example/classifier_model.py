import torch

from llm_example.gpt_example.gpt_model.load_gpt_model import load_gpt_model

def get_classifier_model(device):
    model, config = load_gpt_model(device=device)

    # 首先冻结模型，即将所有层设为不可训练
    for param in model.parameters():
        param.requires_grad = False

    # 添加分类层
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=config["emb_dim"],
        out_features=num_classes
    )

    # 使最终层归一化和最后一个Transformer块可训练
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model, config

def get_trained_classifier_model(device):
    model, config = get_classifier_model(device)
    model_state_dict = torch.load("data/output/review_classifier.pth", map_location=device)
    model.load_state_dict(model_state_dict)
    
    return model, config
