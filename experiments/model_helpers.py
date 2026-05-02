import torch

def train_model(num_epochs, loader, model, loss_fn, optimizer, config):
    device = model.device
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x).logits
            loss = loss_fn(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  step {step}/{len(loader)}  loss={loss.item():.4f}")
        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")


def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device="cuda"):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0].tolist())