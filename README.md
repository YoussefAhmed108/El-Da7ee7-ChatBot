# El-Da7ee7-ChatBot

This repository contains code for fine-tuning a text generation model to produce output in Egyptian dialect Arabic, specifically mimicking the style of the famous presenter El Da7ee7. The system incorporates Retrieval-Augmented Generation (RAG) to ground the generated responses in factual context retrieved from sources such as Wikipedia. **Note:** This project is still under development and has not yet been deployed.

---

## Table of Contents

1. [Data Collection]((#1-data-collection))
2. [Data Preprocessing](#2-data-preprocessing)
3. [Retrieval-Augmented Generation (RAG)](#3-retrieval-augmented-generation-rag)
4. [Model Training](#4-model-training)
5. [Future Work & Deployment](#5-future-work--deployment)



## 1. Data Collection

In this phase, we collect the transcripts required for fine-tuning our model. We utilize the YouTube API to retrieve videos from various Egyptian channels, and then use the [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) to fetch the transcripts for those videos. The collected transcripts, along with their titles and relevant metadata, are stored in a JSON file called `Transcripts.json`. Additionally, we filter out and discard any videos that are less than 5 minutes long to ensure that our dataset contains sufficiently detailed content.

**Key Points:**
- Videos are retrieved from various Egyptian channels using the YouTube API.
- Transcripts are fetched using the youtube-transcript-api.
- Transcripts, titles, and metadata are saved in `Transcripts.json`.
- Videos with a duration of less than 5 minutes are discarded.
- The final dataset comprises approximately 700 transcripts, including around 150 transcripts from El Da7ee7’s videos, ensuring a strong representation of his unique style and dialect.


## 2. Data Preprocessing

In this phase, we take the raw transcripts collected from YouTube and transform them into a clean, uniform dataset for fine-tuning our model. Our preprocessing pipeline consists of the following steps:

1. **Title and Transcript Cleaning:**  
   - **Title Cleaning:** We remove extraneous parts from video titles (e.g., filtering out segments containing `"الدحيح"`) using the `clean_title` function.  
   - **Transcript Cleaning:** We remove newline characters and literal backslashes from the transcripts using the `clean_transcript` function.

2. **Data Transformation and Filtering:**  
   - Transcripts, along with their titles and metadata, are stored in a JSON file named `Transcripts.json`.  
   - Only transcripts from videos longer than 5 minutes are retained.  
   - Based on the video title, we prepend a style instruction to the transcript. For example, if the title indicates that the video is from El Da7ee7, we prepend:  
     ```
     تَحَدَّث كأنك 'الدحيح'
     ```  
     Otherwise, we prepend:  
     ```
     استخدم اللهجة المصرية الدارجة مع تعابير شعبية
     ```

3. **Arabic Normalization:**  
   - We normalize the text by removing diacritics, Tatweel (kashida), and normalizing different forms of the letter Alif.  
   - Punctuation is removed and the text is lowercased to ensure consistency.

4. **Combining Title and Transcript:**  
   - The cleaned title and transcript are combined into a single string (with a separator) using the `combine_text` function.  
   - The final output is stored in a new column called `"processed"` in the DataFrame, which is then used for tokenization and further training.


## 3. Retrieval-Augmented Generation (RAG)

To improve the factual grounding of our generated text, we integrate a Retrieval-Augmented Generation (RAG) component into our pipeline. This component retrieves relevant external context and embeds it directly into the prompt, serving as context ingestion for the generation model. We use two complementary methods to gather this context:

### 3.1 Tavily-Based Web Search

We leverage the Tavily API to perform real-time web searches based on the user’s prompt (written in Egyptian Arabic). This method returns search results—including URLs, titles, and snippets—that provide up-to-date external context. The retrieved URLs can be followed to extract additional text if needed, though in our current implementation we primarily use the snippets as part of our context.

### 3.2 Keyword Extraction and Wikipedia Passage Retrieval

In parallel, we use a combination of KeyBERT and RAKE for robust keyword extraction:
- **Keyword Extraction:**  
  We extract candidate keywords from the user prompt using both KeyBERT and RAKE. These candidates are then combined using a voting system that includes normalization, stopword filtering, and even filtering out common verbs. This ensures that the most relevant terms for the topic are selected.
  
- **Wikipedia Passage Retrieval:**  
  The extracted keywords are used to query Arabic Wikipedia via its API. For each keyword, we fetch page extracts and titles. The results are then embedded by:
  - **Embedding Retrieval:**  
    We compute embeddings for the fetched passages using an Arabic SentenceTransformer.
  - **FAISS Indexing (Optional):**  
    We build a FAISS index to efficiently rank and retrieve the top-k passages based on semantic similarity with the original prompt.

### 3.3 Context Ingestion

Once the relevant passages are retrieved:
- **Concatenation:**  
  The retrieved passages (or their excerpts) are concatenated together.
- **Prompt Augmentation:**  
  This concatenated context is then appended to the original user prompt. This enriched prompt serves as input to our generation model, ensuring that the model has direct access to the factual context during text generation. This technique is referred to as "context ingestion," as the model ingests external factual data as part of its prompt.


## 4. Model Training

In this phase, we fine-tune an Arabic GPT-2–style model (using AraGPT2) on our preprocessed dataset. Our goal is to adapt the model to generate text in Egyptian Arabic, incorporating the unique style of El Da7ee7. We leverage several techniques to manage GPU memory and prevent overfitting:

- **Data Collation for Causal LM:**  
  We use a `DataCollatorForLanguageModeling` configured with `mlm=False` since we are training a causal language model.

- **Mixed Precision Training:**  
  Enabling FP16 (mixed precision) reduces memory usage and speeds up training.

- **Gradient Accumulation:**  
  With a per-device batch size of 1, we use gradient accumulation (`gradient_accumulation_steps=4`) to simulate a larger effective batch size without exceeding GPU memory limits.

- **Early Stopping:**  
  An `EarlyStoppingCallback` is used to halt training when the validation loss stops improving (with a patience of 2 epochs). In addition, the Trainer is configured to load the best model at the end of training based on evaluation loss.

- **Checkpointing and Saving:**  
  Checkpoints are saved at the end of each epoch. The final best model and tokenizer are then saved to a designated folder for later inference or further fine-tuning.


## 5. Future Work & Deployment

This project is still under active development, particularly in the model training phase. Our ultimate goal is to create a robust model that generates text in Egyptian dialect Arabic in the unique style of El Da7ee7, but there are some current challenges and planned improvements:

### Current Challenges
- **Limited Computational Resources:**  
  Fine-tuning large models like AraGPT2 (especially the mega variant) requires substantial GPU resources. Our current setup is constrained by limited GPU availability, which affects both training time and model performance.
  
- **Data Availability:**  
  Although our dataset comprises approximately 700 transcripts, including around 150 from El Da7ee7, this amount of data is still relatively limited for capturing the full diversity and nuances of Egyptian Arabic. Additional high-quality, in-domain data is needed to further improve the model’s performance and generalization.

- **Model Training:**  
  Early experiments indicate that the model starts to overfit after a few epochs. We are exploring techniques like gradient accumulation, mixed precision training, advanced regularization, and even alternative architectures to better capture the desired style without overfitting.

### Planned Improvements
- **Enhanced Training Strategies:**  
  We plan to experiment with different training techniques (e.g., deeper hyperparameter tuning, advanced regularization methods, and leveraging DeepSpeed/ZeRO optimizations) to efficiently train larger models with our available GPU resources.
  
- **Data Augmentation & Collection:**  
  We are working on expanding our dataset by collecting more transcripts from Egyptian Arabic sources and exploring data augmentation techniques that preserve the unique dialect and style.
  
- **Deployment of a Chat Interface:**  
  In addition to further fine-tuning the model, we are developing a web-based chat interface where users can interact with the model in real time you can check its repo [here](https://github.com/YoussefAhmed108/El-Da7ee7-ChatBot-Website). This chat interface will allow users to experience the model's Egyptian dialect and style directly.

### Contributions
 We welcome contributions, feedback, and suggestions from the community to help overcome current challenges and improve the system.

*Note:* This project is still under development and is not yet deployed. Future iterations will address the challenges mentioned above and aim to deliver a robust, production-ready solution.