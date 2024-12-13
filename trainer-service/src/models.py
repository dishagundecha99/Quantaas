from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import torch

class LoadModel():
    '''
    LoadModel class to load the type of model into memory
    '''
    def __init__(self, device, tokenizer_name, model_name, task_type="classification") -> None:
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        if task_type == "classification":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.sentiment_analysis_primer()
        elif task_type == "questionanswering":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.question_answering_primer()
        elif task_type == "machinetranslation":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.machine_translation_primer()
        elif task_type == "summarization":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.summarization_primer()
        else:
            print("None of the NLP tasks selected!")
            pass

    def sentiment_analysis_primer(self):
        input_text = "Well, he is going to love it!"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        print(f"Predicted Sentiment Class: {predicted_class}\nSuccessfully primed the model")

    def question_answering_primer(self):
        question = "On which date did Swansea City play its first Premier League game?"
        context = "In 2011, a Welsh club participated in the Premier League for the first time after Swansea City gained promotion. \
        The first Premier League match to be played outside England was Swansea City's home match at the Liberty Stadium against Wigan Athletic on 20 August 2011. \
        In 2012\u201313, Swansea qualified for the Europa League by winning the League Cup. The number of Welsh clubs in the Premier League increased to two for the first time in 2013\u201314, \
        as Cardiff City gained promotion, but Cardiff City was relegated after its maiden season."

        inputs = self.tokenizer(question, context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        print(f"Answer: {self.tokenizer.decode(predict_answer_tokens)}\nSuccessfully primed the model")

    def machine_translation_primer(self):
        text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. \
        It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. \
        It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=100, do_sample=False)
        print(f"Summarized text: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\nSuccessfully primed the model")

    def summarization_primer(self):
        text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. \
        It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. \
        It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=100, do_sample=False)

        print(f"Summarized text: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\nSuccessfully primed the model")

    def get_model(self):
        return self.tokenizer, self.model
    
'''from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import torch
import torch.quantization

class LoadModel():

    def __init__(self, device, tokenizer_name, model_name, task_type="classification") -> None:
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

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
            print("None of the NLP tasks selected!")

        print(f"Loaded {task_type} model: {model_name}")

    def prune_model(self):
        # Simple pruning approach (example: pruning a specific layer or parameter)
        print("Pruning model...")
        self.model.eval()
        torch.nn.utils.prune.l1_unstructured(self.model.encoder.layer[0].attention.self.query, name="weight", amount=0.2)

    def quantize_model(self):
        # Apply quantization
        print("Quantizing model...")
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    def evaluate_model(self, tokenized_train, tokenized_test):
        print("Evaluating model...")
        # Evaluate original, pruned, and quantized models
        results = {}
        
        # Evaluating Original Model
        original_accuracy = self.evaluate(tokenized_train, tokenized_test)
        results["original"] = original_accuracy
        
        # Evaluate Pruned Model
        self.prune_model()
        pruned_accuracy = self.evaluate(tokenized_train, tokenized_test)
        results["pruned"] = pruned_accuracy
        
        # Evaluate Quantized Model
        self.quantize_model()
        quantized_accuracy = self.evaluate(tokenized_train, tokenized_test)
        results["quantized"] = quantized_accuracy
        
        print(f"Evaluation results: {results}")
        return results

    def evaluate(self, tokenized_train, tokenized_test):
        # Use a dummy evaluation for now (can be replaced with actual evaluation logic)
        # This can include training loop, accuracy calculation, etc.
        accuracy = 0.95  # Dummy accuracy value
        return accuracy
    
    def get_model(self):
        return self.tokenizer, self.model
'''