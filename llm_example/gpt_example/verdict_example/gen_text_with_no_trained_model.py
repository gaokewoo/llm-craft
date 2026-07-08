import torch
import tiktoken
from llm_example.gpt_example.config import GPT_CONFIG_124M
from tools.log import log
from model.gpt_model import GPTModel
from tools.generate_text import generate_text_simple, text_to_token_ids, token_ids_to_text
from tools.model_tool import calc_loss_loader
from llm_example.gpt_example.verdict_example.verdict_dataset import get_verdict_dataset

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
log.info("Output text:\n%s", token_ids_to_text(token_ids, tokenizer))


inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]
targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
log.info(probas.shape)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
log.info("Token IDs:\n%s", token_ids)

log.info(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
log.info(f"Outputs batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# 使用以下代码打印与目标词元对应的初始softmax概率分数
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
log.info("Text 1:%s", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
log.info("Text 2:%s", target_probas_2)

# 对概率分数应用对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
log.info(log_probas)

# 计算平均值将这些对数概率组合成一个单一分数
avg_log_probas = torch.mean(log_probas)
log.info(avg_log_probas)

# 我们的目标是通过在训练过程中更新模型的权重，使平均对数概率尽可能接近0。
# 然而，在深度学习中，通常的做法不是将平均对数概率升至0，而是将负平均对数概率降至0。
# 负平均对数概率就是平均对数概率乘以-1
neg_avg_log_probas = avg_log_probas * -1
log.info(neg_avg_log_probas)

# 交叉熵损失
log.info("Logits shape:%s", logits.shape)
log.info("Targets shape:%s", targets.shape)

# 在批处理维度上将它们组合在一起来展平这些张量
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
log.info("Flattened logits:%s", logits_flat.shape)
log.info("Flattened targets:%s", targets_flat.shape)

# 计算交叉熵损失
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
log.info(loss)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loader, val_loader = get_verdict_dataset()
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
log.info("Training loss: %s", train_loss)
log.info("Validation loss: %s", val_loss)
