from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

descriptions = [
    "Add two numbers",
    "Subtract two numbers",
    # Add the rest of your descriptions here...
]

codes = [
    "class Calculator:\n    def add(self, number1, number2):\n        try:\n            return number1 + number2\n        except Exception as e:\n            return str(e)",
    "class Calculator:\n    def subtract(self, number1, number2):\n        try:\n            return number1 - number2\n        except Exception as e:\n            return str(e)",
    # Add the rest of your codes here...
]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

input_tokens = tokenizer(descriptions, padding='max_length', truncation=True, max_length=100, return_tensors="pt")
target_tokens = tokenizer(codes, padding='max_length', truncation=True, max_length=100, return_tensors="pt")


class CodeGenDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels['input_ids'])


dataset = CodeGenDataset(input_tokens, target_tokens)
print(dataset.labels)
print(dataset.encodings)


model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=10,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Define a code-related prompt
prompt = """
[INST]
You are an expert Python programmer and personal assistant, here is your task: add 2 numbers
Your answer should start with a [PYTHON] tag and end with a [/PYTHON] tag.
[/INST]
"""

# Tokenize the prompt with attention_mask
input_encodings = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=100, truncation=True)

# Generate code using the model
generated_ids = model.generate(
    input_ids=input_encodings['input_ids'],
    attention_mask=input_encodings['attention_mask'],
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated tokens to a string
generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(generated_code)
