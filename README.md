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
@inproceedings{shah-etal-2023-trillion,
    title = "Trillion Dollar Words: A New Financial Dataset, Task {\&} Market Analysis",
    author = "Shah, Agam  and
      Paturi, Suvan  and
      Chava, Sudheer",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.368",
    doi = "10.18653/v1/2023.acl-long.368",
    pages = "6664--6679",
    abstract = "Monetary policy pronouncements by Federal Open Market Committee (FOMC) are a major driver of financial market returns. We construct the largest tokenized and annotated dataset of FOMC speeches, meeting minutes, and press conference transcripts in order to understand how monetary policy influences financial markets. In this study, we develop a novel task of hawkish-dovish classification and benchmark various pre-trained language models on the proposed dataset. Using the best-performing model (RoBERTa-large), we construct a measure of monetary policy stance for the FOMC document release days. To evaluate the constructed measure, we study its impact on the treasury market, stock market, and macroeconomic indicators. Our dataset, models, and code are publicly available on Huggingface and GitHub under CC BY-NC 4.0 license.",
}
```

## Contact Information

Please raise issue on GitHub or contact Agam Shah (ashah482[at]gatech[dot]edu) for any issues and questions.  
GitHub: [@shahagam4](https://github.com/shahagam4) 
Website: [https://shahagam4.github.io/](https://shahagam4.github.io/)



