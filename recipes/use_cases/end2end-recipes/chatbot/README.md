## Introduction

Large language models (LLMs) have emerged as groundbreaking tools, capable of understanding and generating human-like text. These models power many of today's advanced chatbots, providing more natural and engaging user experiences. But how do we create these intelligent systems?

Here, we aim to make an FAQ model for Llama that be able to answer questions about Llama by fine-tune Llama2 7B chat using existing official Llama documents.


### Fine-tuning Process

Fine-tuning LLMs here LLama 2 involves several key steps: Data Collection, preprocessing, fine-tuning, evaluation.


### LLM Generated datasets

As Chatbots are usually domain specifics and based on public or proprietary data, one common way inspired by [self-instruct paper](https://arxiv.org/abs/2212.10560) is to use LLMs to assist building the dataset from our data. For example to build an FAQ model, we can use Llama model to process our documents and help us build question and answer pair (We will showcase this here). Just keep it in mind that usually most of the proprietary LLMs has this clause in their license that you are not allowed to use the output generated from the model to train another LLM. In this case we will use Llama to fine-tune another Llama model.


Similarly, we will use the same LLM to evaluate the quality of generated datasets and finally evaluate the outputs from the model.


Given this context, here we want to highlight some of best practices that need to be in place for data collection and pre-processing in general.

### **Data Collection & Preprocessing:**

Gathering a diverse and comprehensive dataset is crucial. This dataset should include a wide range of topics and conversational styles to ensure the model can handle various subjects. A recent [research](https://arxiv.org/pdf/2305.11206.pdf) shows that quality of data has far more importance than quantity. Here are some high level thoughts on data collection and preprocessing along with best practices:

**NOTE** data collection and processing is very use-case specific and here we can only share best practices but it would be very nuanced for each use-case.

- Source Identification: Identify the sources where your FAQs are coming from. This could include websites, customer service transcripts, emails, forums, and product manuals. Prioritize sources that reflect the real questions your users are asking.

- Diversity and Coverage: Ensure your data covers a wide range of topics relevant to your domain. It's crucial to include variations in how questions are phrased to make your model robust to different wording.

- Volume: The amount of data needed depends on the complexity of the task and the variability of the language in your domain. Generally, more data leads to a better-performing model, but aim for high-quality, relevant data.

Here, we are going to use [self-instruct](https://arxiv.org/abs/2212.10560) idea and use Llama model to build our dataset, for details please check this [doc](./data_pipelines/REAME.md).


**Things to keep in mind**

- **Pretraining Data as the Foundation**: Pretraining data is crucial for developing foundational models, influencing both their strengths and potential weaknesses. Fine-tuning data refines specific model capabilities and, through instruction fine-tuning or alignment training, enhances general usability and safety.

- **Quality Over Quantity**: More data doesn't necessarily mean better results. It's vital to select data carefully and perform manual inspections to ensure it aligns with your project's aims.

- **Considerations for Dataset Selection**: Selecting a dataset requires considering various factors, including language and dialect coverage, topics, tasks, diversity, quality, and representation.

- **Impact of Implicit Dataset Modifications**: Most datasets undergo implicit changes during selection, filtering, and formatting. These preprocessing steps can significantly affect model performance, so they should not be overlooked.

- **Finetuning Data's Dual-Edged Sword**: Finetuning can improve or impair model capabilities. Make sure you know the nature of your data to make an informed selections.

- **Navigating Dataset Limitations**: The perfect dataset for a specific task may not exist. Be mindful of the limitations when choosing from available resources, and understand the potential impact on your project.

#### **Best Practices for FineTuning Data Preparation**

- **Enhancing Understanding with Analysis Tools**: Utilizing tools for searching and analyzing data is crucial for developers to gain a deeper insight into their datasets. This understanding is key to predicting model behavior, a critical yet often overlooked phase in model development.

- **The Impact of Data Cleaning and Filtering**: Data cleaning and filtering significantly influence model characteristics, yet there's no universal solution that fits every scenario. Our guidance includes filtering recommendations tailored to the specific applications and communities your model aims to serve.

- **Data Mixing from Multiple Sources**: When training models with data from various sources or domains, the proportion of data from each domain (data mixing) can greatly affect downstream performance. It's a common strategy to prioritize "high-quality" data domains—those with content written by humans and subjected to an editing process, like Wikipedia and books. However, data mixing is an evolving field of research, with best practices still under development.

- **Benefits of Removing Duplicate Data**: Eliminating duplicated data from your dataset can lessen unwanted memorization and enhance training efficiency.

- **The Importance of Dataset Decontamination**: It's crucial to meticulously decontaminate training datasets by excluding data from evaluation benchmarks. This ensures the model's capabilities are accurately assessed.


**Data Exploration and Analysis**

- Gaining Insights through Dataset Exploration: Leveraging search and analysis tools to explore training datasets enables us to cultivate a refined understanding of the data's contents, which in turn influences the models. Direct interaction with the data often reveals complexities that are challenging to convey or so might not be present in the documents.

- Understanding Data Complexity: Data, especially text, encompasses a wide array of characteristics such as length distribution, topics, tones, formats, licensing, and diction. These elements are crucial for understanding the dataset but are not easily summarized without thorough examination.

- Utilizing Available Tools: We encourage to take advantage of the numerous tools at your disposal for searching and analyzing your training datasets, facilitating a deeper comprehension and more informed model development.

**Tools**

- [wimbd](https://github.com/allenai/wimbd) for data analysis.
- TBD



**Data Cleaning**

Purpose of Filtering and Cleaning: The process of filtering and cleaning is essential for eliminating unnecessary data from your dataset. This not only boosts the efficiency of model training but also ensures the data exhibits preferred characteristics such as high informational value, coverage of target languages, low levels of toxicity, and minimal presence of personally identifiable information.

Considering Trade-offs: We recommend to carefully weigh the potential trade-offs associated with using certain filters, it may impact the diversity of your data, [removing minority individuals](https://arxiv.org/abs/2104.08758).

**Tools**
- [OpenRefine](https://github.com/OpenRefine/OpenRefine?tab=readme-ov-file),(formerly Google Refine): A standalone open-source desktop application for data cleanup and transformation to other formats. It's particularly good for working with messy data, including data format transformations and cleaning.

- [FUN-Langid](https://github.com/google-research/url-nlp/tree/main/fun-langid), simple, character 4-gram LangID classifier recognizing up to 1633 languages.

- Dask: Similar to Pandas, Dask is designed for parallel computing and works efficiently with large datasets. It can be used for data cleaning, transformations, and more, leveraging multiple CPUs or distributed systems.




**Data Deduplication**

- **Data Deduplication importance**: Data deduplication is a important preprocessing step to eliminate duplicate documents or segments within a document from the dataset. This process helps in minimizing the model's chance of memorizing unwanted information, including generic text, copyrighted content, and personally identifiable details.

- **Benefits of Removing Duplicates**: Aside from mitigating the risk of undesirable memorization, deduplication enhances training efficiency by decreasing the overall size of the dataset. This streamlined dataset contributes to a more effective and resource-efficient model training process.

- **Assessing the Impact of Duplicates**: You need to carefully evaluate the influence of duplicated data on their specific model use case. Memorization may be beneficial for models designed for closed-book question answering, or similarly chatbots.

**Tools**

- [thefuz](https://github.com/seatgeek/thefuzz): It uses Levenshtein Distance to calculate the differences between sequences in a simple-to-use package.
- [recordlinkage](https://github.com/J535D165/recordlinkage): It is modular record linkage toolkit to link records in or between data sources.

**Data Decontamination**

The process involves eliminating evaluation data from the training dataset. This crucial preprocessing step maintains the accuracy of model evaluation, guaranteeing that performance metrics are trustworthy and not skewed.

**Tools**
- TBD




### **LLama FAQ Use-Case**


1. **Data Collection**
Here, we are going to use self-instruct idea and use Llama model to build our dataset, for details please check this [doc](./data_pipelines/REAME.md).

2. **Data Formatting**

For a FAQ model, you need to format your data in a way that's conducive to learning question-answer relationships. A common format is the question-answer (QA) pair:

Question-Answer Pairing: Organize your data into pairs where each question is directly followed by its answer. This simple structure is highly effective for training models to understand and generate responses. For example:

```python
"question": "What is Llama 2?",
"answer": "Llama 2 is a collection of pretrained and fine-tuned large language models ranging from 7 billion to 70 billion parameters, optimized for dialogue use cases."
```


3. **Preprocessing:** This step involves cleaning the data and preparing it for training. It might include removing irrelevant information, correcting errors, and splitting the data into training and evaluation sets.


4. **Fine-Tuning:** Given that we have a selected pretrained model, in this case we use LLama 2 chat 7B, fine-tunning with more specific data can improve its performance on particular tasks, such as answering questions about Llama in this case.
#### Building Dataset 

During the self-instruct process of generation Q&A pairs from documents, we realized that with out system prompt being
```python
You are a language model skilled in creating quiz questions.
You will be provided with a document,
read it and generate question and answer pairs
that are most likely be asked by a use of llama that just want to start, 
please make sure you follow those rules,
1. Generate only {total_questions} question answer pairs.
2. Generate in {language}.
3. The questions can be answered based *solely* on the given passage. 
4. Avoid asking questions with similar meaning.
5. Make the answer as concise as possible, it should be at most 60 words.
6. Provide relevant links from the document to support the answer.
7. Never use any abbreviation.
8. Return the result in json format with the template: 
  [
    {{
      "question": "your question A.",
      "answer": "your answer to question A."
    }},
    {{
      "question": "your question B.",
      "answer": "your answer to question B."
    }}
  ]

```

Model tends to ignore providing the bigger picture in the questions, for example below is the result of Q&A pair from reading Code Llama paper. Partially, its because due to context window size of the model we have to divide the document into smaller chunks, so model use `described in the passage` or `according to the passage?` in the question instead of linking it back to Code Llama.


```python
{
        "question": "What is the purpose of the transformation described in the passage?",
        "answer": "The transformation is used to create documents with a prefix, middle part, and suffix for infilling training."
    },
{
    "question": "What is the focus of research in transformer-based language modeling, according to the passage?",
    "answer": "The focus of research is on effective handling of long sequences, specifically extrapolation and reducing the quadratic complexity of attention passes."
},
```


#### Data Insights

We generated a dataset of almost 650 Q&A pairs from some of the open source documents about Llama 2, including getting started guide from Llama website, its FAQ, Llama 2, Purple Llama, Code Llama papers and Llama-Recipes documentations. 

We have run some fine-tuning experiments with single GPU using quantization with different LORA configs (all linear layer versus query and key projections only) and different number of epochs. Although train and eval loss shows decrease specially with using all linear layers in LORA configs and training with 6 epochs, still the result is far from acceptable in real tests.


Here is how losses between three runs looks like.

<p align="center">
  <img src=./eval-loss-3runs.png alt="Eval Loss" width="48%" style="margin-right: 2%;"/>
  <img src=./train-loss-3runs.png alt="Train Loss" width="48%"/>
</p>

##### Low Quality Dataset

Below are some examples of real test on the fine-tuned model with very poor results. It seems fine-tuned model does not show any promising results with this dataset. Looking at the dataset, we could observe that the amount of data (Q&A pair) for each concept such as PyTorch FSDP and Llama-Recipe is very limited and almost one pair per concept. This shows lack of relevant training data. The recent research showed that from each taxonomy having 2-3 examples can yield promising results.

<p align="center">
  <img src=./poor-test-1.png alt="Poor Test Results example 1" width="48%" style="margin-right: 2%;"/>
  <img src=./poor-test-2.png alt="Poor Test Results example 1" width="48%"/>
</p>


Next, we are looking into augmenting our datasets. One way to do so, is to use our Llama 70B model to read our question answer pairs and come up with two paraphrase versions of each pair to augment our data. 


