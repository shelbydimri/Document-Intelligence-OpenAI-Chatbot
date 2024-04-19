# import os
# import pandas as pd
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# # Load the dataset
# def load_dataset(file_path):
#     df = pd.read_csv(file_path)
#     dataset = []
#     for index, row in df.iterrows():
#         dataset.append(row["User Input"])
#         dataset.append(row["Chatbot Response"])
#     return dataset

# # Fine-tune the chatbot model
# def fine_tune_chatbot(dataset, model_name_or_path, output_dir):
#     tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
#     model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

#     train_dataset = TextDataset(
#         tokenizer=tokenizer,
#         text_column=dataset,
#         max_length=512,
#         block_size=128
#     )

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#     )

#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=10_000,
#         save_total_limit=2,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#     )

#     trainer.train()
#     trainer.save_model(output_dir)

# # Main function to load dataset and fine-tune the chatbot model
# def main():
#     # Set file paths and output directory
#     dataset_file = "chatbot_dataset.csv"
#     model_name = "gpt-3.5-turbo"
#     output_directory = "fine_tuned_chatbot_model"

#     # Load dataset
#     dataset = load_dataset(dataset_file)

#     # Fine-tune the chatbot model
#     fine_tune_chatbot(dataset, model_name, output_directory)

# if __name__ == "__main__":
#     main()
# >>>>>>>>>>>>>>>>>_--------------2nd

import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import jsonlines

# Load dataset from JSON file
def load_dataset_from_json(json_file):
    dataset = []
    with jsonlines.open(json_file, mode='r') as reader:
        for obj in reader:
            dataset.append(obj['prompt'])
            dataset.append(obj['completion'])
    return dataset

# Fine-tune the chatbot model
def fine_tune_chatbot(dataset, model_name_or_path, output_dir):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        text_column=dataset,
        max_length=512,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

# Main function to load dataset from JSON and fine-tune the chatbot model
def main(json_file):
    # Set model and output directory
    model_name = "gpt-3.5-turbo"
    output_directory = "fine_tuned_chatbot_model"

    # Load dataset from JSON file
    dataset = load_dataset_from_json(json_file)

    # Fine-tune the chatbot model
    fine_tune_chatbot(dataset, model_name, output_directory)

if __name__ == "__main__":
    json_file = "chatbot_training_data.jsonl"
    main(json_file)


# import os
# import pandas as pd
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
# import jsonlines

# # Load dataset from JSON file
# def load_dataset_from_json(json_file):
#     dataset = []
#     with jsonlines.open(json_file, mode='r') as reader:
#         for obj in reader:
#             dataset.append(obj['prompt'])
#             dataset.append(obj['completion'])
#     return dataset

# # Fine-tune the chatbot model
# def fine_tune_chatbot(dataset, model_name_or_path, output_dir):
#     tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
#     model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

#     train_dataset = TextDataset(
#         tokenizer=tokenizer,
#         text_column=dataset,
#         max_length=512,
#         block_size=128
#     )

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#     )

#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=10_000,
#         save_total_limit=2,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#     )

#     trainer.train()
    
#     # Save the fine-tuned model
#     trainer.save_model(output_dir)

# # Main function to load dataset from JSON and fine-tune the chatbot model
# def main(json_file):
#     # Set model and output directory
#     model_name = "gpt-3.5-turbo"
#     output_directory = "fine_tuned_chatbot_model"

#     # Load dataset from JSON file
#     dataset = load_dataset_from_json(json_file)

#     # Fine-tune the chatbot model
#     fine_tune_chatbot(dataset, model_name, output_directory)

# if __name__ == "__main__":
#     json_file = "chat_history.jsonl"
#     main(json_file)
