

# data poisoning
async def local_training(self, epochs=5):
    self.initialize_model_state()
    self.model.train()
    for epoch in range(epochs):
        for inputs, labels in self.data_loader():
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
    delta_w = self.get_model_updates()
    if self.is_malicious: # poisoned here
        for name in delta_w:
            delta_w[name] = delta_w[name] * 10  # exaggerate the updates
    encrypted_updates = self.encrypt_updates(delta_w)
    return encrypted_updates
