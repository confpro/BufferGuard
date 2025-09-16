# Intention-aware Buffer Overflow Vulnerability Detection using LLM-aided Code Information Flow Analysis

# dataset
Because the training set is too large, we have divided it into three parts, and if you need to use it, please splice it together.

# usage of this package
First, You need to go to hugging face to download the original Deepseek-Coder or other models.
In the process folder, csv_AST.py can get IIG, and gpt_process_data.py can get CIF.
In RQ1 folder, the train.py can run the results of BufferGuard.
In RQ2 folder, the column names in dataset.py need to be changed or deleted, center loss can be annotated during training.
In RQ3 folder, the column names in dataset.py need to be changed for each cate.
In RQ4 folder, the model in train.py can be replaced.

# motivation example
<img width="933" height="473" alt="image" src="https://github.com/user-attachments/assets/e4f68d91-b462-4523-abdc-4b6f6b30ee60" />

# Overview of BufferGuard
<img width="882" height="540" alt="image" src="https://github.com/user-attachments/assets/df79b6e4-e8da-4663-bfad-e5cb80c55bc5" />

# An example of CIF extraction.
<img width="1037" height="453" alt="image" src="https://github.com/user-attachments/assets/2002bba8-e971-4cbf-a94a-18bd68afa8ab" />

# RQ1 result
<img width="746" height="807" alt="image" src="https://github.com/user-attachments/assets/577fbaba-9fe0-426c-ab8b-6ab300f10705" />

# RQ2 result
<img width="622" height="342" alt="image" src="https://github.com/user-attachments/assets/e1e2b233-7c4b-4bcc-af4f-20e3eacf0daf" />

# RQ3 result
<img width="618" height="279" alt="image" src="https://github.com/user-attachments/assets/2df233ea-8523-4278-9119-8efd171f44ee" />

# RQ4 result
<img width="888" height="259" alt="image" src="https://github.com/user-attachments/assets/909bff1d-8196-4b7b-b5e4-b93da1bf9aaf" />
