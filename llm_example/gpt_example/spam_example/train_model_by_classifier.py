import tiktoken
import torch
import time

from llm_example.gpt_example.spam_example.spam_dataset import get_spam_dataset
from llm_example.gpt_example.spam_example.classifier_tools import calc_accuracy_loader, classify_review
from llm_example.gpt_example.spam_example.train_classifier_simple import train_classifier_simple
from tools.matplot import plot_values
from tools.log import log
from llm_example.gpt_example.spam_example.classifier_model import get_classifier_model

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config = get_classifier_model(device)

train_loader, val_loader, test_loader = get_spam_dataset(tokenizer)


start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()

execution_time_minutes = (end_time - start_time) / 60
log.info(f"Training completed in {execution_time_minutes:.2f} minutes.")



epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# accuracies are recorded once per epoch, so use a separate tensor
epochs_tensor_acc = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor_acc = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor_acc, examples_seen_tensor_acc, train_accs, val_accs,
    label="accuracy"
)

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

log.info(f"Training accuracy: {train_accuracy*100:.2f}%")
log.info(f"Validation accuracy: {val_accuracy*100:.2f}%")
log.info(f"Test accuracy: {test_accuracy*100:.2f}%")

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

log.info("%s", classify_review(
    text_1, model, tokenizer, device, max_length=train_loader.dataset.max_length
))


text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

log.info("%s", classify_review(
    text_2, model, tokenizer, device, max_length=train_loader.dataset.max_length
))

# 保存模型
torch.save(model.state_dict(), "data/output/review_classifier.pth")
