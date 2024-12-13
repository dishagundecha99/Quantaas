from transformers import DefaultDataCollator, DataCollatorWithPadding, DataCollatorForSeq2Seq
from datasets import load_dataset
import ast

class LoadDataset:
    """
    LoadDataset class used for loading the test dataset into memory.
    """
    def __init__(self, test_path, model, task_type) -> None:
        self.test_dataset = test_path
        self.model = model
        self.task_type = task_type
        self.data_collator = None
        self.tokenized_test = None
        self.process()

    def preprocess_function_sentiment(self, examples):
        return self.model.tokenizer(examples["text"], truncation=True)

    def preprocess_function_qa(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.model.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = ast.literal_eval(answers[i])
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_function_mt(self, examples):
        source_lang = "en"
        target_lang = "fr"
        prefix = "translate English to French: "

        inputs = [prefix + ast.literal_eval(example)[source_lang] for example in examples["translation"]]
        targets = [ast.literal_eval(example)[target_lang] for example in examples["translation"]]
        model_inputs = self.model.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    def preprocess_function_s(self, examples):
        prefix = "summarize: "

        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = self.model.tokenizer(inputs, max_length=1024, truncation=True)
        labels = self.model.tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def process(self):
        try:
            # Load the test dataset
            test_dataset = load_dataset("csv", data_files=self.test_dataset)["train"]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if self.task_type == "classification":
            self.tokenized_test = test_dataset.map(self.preprocess_function_sentiment, batched=True)
            self.data_collator = DataCollatorWithPadding(tokenizer=self.model.tokenizer)
        elif self.task_type == "questionanswering":
            self.tokenized_test = test_dataset.map(self.preprocess_function_qa, batched=True)
            self.data_collator = DefaultDataCollator()
        elif self.task_type == "machinetranslation":
            self.tokenized_test = test_dataset.map(self.preprocess_function_mt, batched=True)
            self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.model.tokenizer, model=self.model.model_name)
        elif self.task_type == "summarization":
            self.tokenized_test = test_dataset.map(self.preprocess_function_s, batched=True)
            self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.model.tokenizer)
        else:
            print("None of the NLP tasks selected!")
            return

        print("Safely loaded and tokenized dataset.")
