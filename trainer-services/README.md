# Trainer method

We give the option for the user to train the following 4 types of NLP problems:

1. Sentiment Classification
2. Question Answering
2. Summarization
3. Machine Translation

We expose three classes:

1. Models
2. Datasets
3. Trainer

All three classes needs to be used to train and save a model. The procedure to run the micro-service is 1) Load model 2) Load Dataset 3) Send to Trainer to fine-tune and save if needed. Further we expose the following methods per class. 

### Models

This class is used for loading a user defined model and tokenizer into memory. This model will be a HuggingFace model and its corresponding tokenizer name. The `__init__` method we will initialize the model requested. Sample usage:

```
device = "cuda:0"
task_type="summarization"
tokenizer_name = "stevhliu/my_awesome_billsum_model"
model_name = "stevhliu/my_awesome_billsum_model"

model_c = LoadModel(device=device,
                    tokenizer_name=tokenizer_name,
                    model_name=model_name,
                    task_type=task_type)
```

This will load the model and prime it i.e. run a sample model inference to check if it is working.

### Datasets

This class is used to load the training, test, or validation dataset used for fine-tuning the HuggingFace model. The `__init__` method reads the data sent and preprocesses it according to the tokenizer created in the previous model class. Therefore, the `model_c` class is sent as an argument to this class creation. Sample usage:

```
train_path="./data/train_summarization.csv"
test_path="./data/test_summarization.csv"

dataset = LoadDataset(train_path=train_path,
                      test_path=test_path,
                      model=model_c,
                      task_type=task_type)
```

### Trainer

This is the main fine-tuning class where we expose the `__init__`, `train`, `evaluate`, and `save` methods. This can be used to perform the tasks as needed and the previous two dataset and model classes needs to be sent as a parameter to this class. Sample usage:

```
repo_name = "finetuning-s-model-3000-samples"
hyperparams = "./data/hyperparameters.json"
save_path = "./output"
user_name = "s1"

tokenizer, model = model_c.get_model()

trainer = CustomTrainer(repo_name=repo_name,
                        hyperparameters_path=hyperparams,
                        tokenizer=tokenizer,
                        model=model,
                        dataset=dataset)

trainer.train() # Calls to train the model, this process takes time
trainer.evaluate() # Calls to evaluate on test set

trainer.save(save_path=save_path,
             user_name=user_name)
```

The sample `demo.py` code can also be run to execute the entire step with approriate parameters. Some example models that we have tested on are as follows:

#### SENTIMENT CLASSIFICATION
https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis

```
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis").to(device)
```

#### QUESTION ANSWERING
https://huggingface.co/Falconsai/question_answering_v2

```
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("Falconsai/question_answering_v2").to(device)
```

#### MACHINE TRANSLATION
https://huggingface.co/t5-small

```
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

#### SUMMARIZATION
https://huggingface.co/stevhliu/my_awesome_billsum_model

```
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
```