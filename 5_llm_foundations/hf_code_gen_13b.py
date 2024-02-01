from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-hf")

# Define a code-related prompt
prompt = "Write a Python function to add two numbers:"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate code using the model
generated_ids = model.generate(input_ids, max_length=100)

# Decode the generated tokens to a string
generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(generated_code)
