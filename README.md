# Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis

This codebase contains the python scripts for the model, dataset, and market analysis for the ACL 2023 paper, "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis". This work was done at the Financial Services Innovation Lab of Georgia Tech. The FinTech lab is a hub for finance education, research and industry in the Southeast. 

The paper is available at [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4447632) 

## Codebase

This codebase contains the python scripts for "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis".

### Environment and setup
Conda environment can be set up using the ``` environment.yml ``` file


### How to use
```
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
```

### Label Interpretation
LABEL_2: Neutral  
LABEL_1: Hawkish  
LABEL_0: Dovish 


## Model
The fine-tuned version of the best performing model 'RoBERTa-large' is available here on Huggingface: [gtfintechlab/FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa)


## Datasets
All the annotated datasets with train-test splits for 3 seeds are available in [GitHub folder](https://github.com/gtfintechlab/fomc-hawkish-dovish/tree/main/training_data/test-and-training)

## Cite
Please cite our paper if you use any code, data, or models.

```c
@article{shah2023trillion, 
  title={Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis},
  author={Shah, Agam and Paturi, Suvan and Chava, Sudheer},
  journal={Available at SSRN 4447632},
  year={2023}
}
```

## Contact Information

Please raise issue on GitHub or contact Agam Shah (ashah482[at]gatech[dot]edu) for any issues and questions.  
GitHub: [@shahagam4](https://github.com/shahagam4) 
Website: [https://shahagam4.github.io/](https://shahagam4.github.io/)



