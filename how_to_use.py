from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa", do_lower_case=True, do_basic_tokenize=True)

model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa", num_labels=3)

config = AutoConfig.from_pretrained("gtfintechlab/FOMC-RoBERTa")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, config=config, device=0, framework="pt")
results = classifier(["Such a directive would imply that any tightening should be implemented promptly if developments were perceived as pointing to rising inflation.", 
                      "The International Monetary Fund projects that global economic growth in 2019 will be the slowest since the financial crisis."], 
                      batch_size=128, truncation="only_first")

print(results)