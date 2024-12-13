from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import torch

class LoadModel():
    '''
    LoadModel class to load the type of model into memory.
    '''
    def __init__(self, device, tokenizer_name, model_name, task_type="classification") -> None:
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        # Load model based on task type
        if task_type == "classification":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        elif task_type == "questionanswering":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        elif task_type == "machinetranslation":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        elif task_type == "summarization":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError(f"Task type '{task_type}' not supported.")
        
        print(f"Loaded {task_type} model: {model_name}")

    def get_model(self):
        return self.tokenizer, self.model

