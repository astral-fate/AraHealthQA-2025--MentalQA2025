# about AraHealthQA 2025

Part of the The Third Arabic Natural Language Processing Conference
(ArabicNLP 2025) Co-located with EMNLP 2025

# Shared Task Details

Track 1: MentalQA 2025

# Motivation

The motivation for this shared task is rooted in the growing global recognition of mental health's importance and the escalating demand for accessible and reliable support systems. This need is particularly pronounced within the Arabic NLP landscape, where specific linguistic and cultural nuances present unique challenges and opportunities. Mental health resources often lack representation in languages other than English, and while efforts exist for other languages, Arabic remains understudied in this domain. 

In particular, the motivations for the shared task are twofold: Social and NLP-specific. 

Socially, there is an urgent and critical need for mental health support in the Arab world. While mental health concerns are pressing globally, Arabic-speaking communities face a pronounced shortage of culturally and linguistically tailored resources. This scarcity severely restricts access to essential information and support. By facilitating the creation of Arabic mental health resources, this shared task aims to bridge this critical gap. Specifically, the task promotes the development of effective, accurate, and culturally appropriate NLP tools for mental health. Such tools can empower individuals by reducing stigma, encouraging them to seek help, and enabling informed decisions regarding their well-being. Ultimately, these resources can significantly enhance mental health outcomes and generate meaningful positive social impact within Arabic-speaking communities.

From an NLP perspective, the shared task also aims to stimulate innovation and drive progress within the Arabic NLP field. By generating new resources and modeling opportunities, this task will encourage advancements across several NLP domains. Mental health presents unique linguistic and semantic challenges, thereby fostering developments in information retrieval (finding relevant and reliable information), semantic understanding (accurately interpreting meaning and intent), and answer generation (producing precise, informative, and culturally appropriate responses). By emphasizing accuracy, reliability, and cultural sensitivity, this shared task will contribute to the creation of more sophisticated and robust Arabic NLP systems, significantly advancing the field.


# Dataset

This shared task leverages the MentalQA dataset, a newly constructed Arabic Question Answering dataset specifically designed for the mental health domain. MentalQA comprises a total of 500 questions and answers (Q&A) posts, including both question types and answer strategies.  For the purpose of the shared task, the dataset will be divided into three subsets: 300 samples for training, 50 samples for development (Dev), and 150 samples for testing, and 150 for blind test. 

The training set can be used to fine-tune large language models (LLMs) or serve as a base for few-shot learning approaches. The development set is intended for tuning model hyperparameters and evaluating performance, while the test set will be used for final evaluation to ensure fair benchmarking of models across participants. 


 The question categories include (Q): 

(A) Diagnosis (questions about interpreting clinical findings) 

(B) Treatment (questions about seeking treatments)

(C) Anatomy and Physiology (questions about basic medical knowledge)

(D) Epidemiology (questions about the course, prognosis, and etiology of diseases) 

(E) Healthy Lifestyle (questions related to diet, exercise, and mood control) 

(F) Provider Choices (questions seeking recommendations for medical professionals and facilities).

(Z). Other. Questions that do not fall under the above-mentioned categories.

The answer strategies include (A): 

(1) Information (answers providing information, resources, etc.)

(2) Direct Guidance (answers providing suggestions, instructions, or advice) 

(3) Emotional Support (answers providing approval, reassurance, or other forms of emotional support). 

## Tasks: The first two sub-tasks involve multi-label classification, while the third is a text generation task.

- Sub-Task-1: Question Categorization: The first task focuses on classifying patient questions into one or more specific types (e.g., diagnosis, treatment, anatomy, etc.) to facilitate a deeper understanding of the user’s intent. This categorization enables the development of intelligent question-answering systems tailored to address diverse patient needs.

- Sub-Task-2: Answer Categorization: The second task centers on classifying answers according to their strategies (e.g., informational, direct guidance, emotional support), ensuring the extraction of relevant and contextually appropriate information.



## Evaluation:

For Sub-Task 1 and Sub-Task 2 — both formulated as multi-label classification tasks — we will use weighted F1 and the Jaccard score. 

# citation

For additional information, refer to the original work of MentalQA:

Alhuzali, H., Alasmari, A., & Alsaleh, H. (2024). MentalQA: An Annotated Arabic Corpus for Questions and Answers of Mental Healthcare. IEEE Access.

Link: https://ieeexplore.ieee.org/document/10600466 
