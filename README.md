# Semantic Chunking Service

A lightweight, production-ready semantic text chunking service built with FastAPI and LangChain's SemanticChunker, designed for deployment on render.com's free tier and seamless integration with n8n workflows.

## Features

- **Semantic Chunking**: Uses LangChain's SemanticChunker with OpenAI embeddings to intelligently split text based on semantic similarity
- **n8n Integration**: REST API endpoints optimized for n8n workflow automation
- **Memory Optimized**: Designed for render.com's free tier (256MB RAM, 0.1 CPU)
- **Caching System**: LRU cache for improved performance and reduced API costs
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Batch Processing**: Support for processing multiple texts in a single request
- **Health Monitoring**: Comprehensive health checks and logging

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/leex279/dynamous-semantic-chunking-service.git
   cd dynamous-semantic-chunking-service
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the service**
   ```bash
   python main.py
   ```

The service will be available at `http://localhost:8000`

### Deploy to Render.com

1. **Fork this repository**
2. **Connect to Render.com**
3. **Add environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `API_KEYS`: Your API keys in format `key1:user1,key2:user2`
   - `ALLOWED_ORIGINS`: Comma-separated list of allowed origins (optional)
4. **Deploy using the included render.yaml**

## API Endpoints

### POST `/api/chunk`
Chunk a single text into semantic segments.

**Headers:**
```
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

**Request:**
```json
{
  "text": "Your text to chunk here...",
  "breakpoint_threshold_type": "percentile",
  "breakpoint_threshold_amount": 95
}
```

**Response:**
```json
{
  "chunks": ["First semantic chunk...", "Second semantic chunk..."],
  "metadata": {
    "total_chunks": 2,
    "original_length": 500,
    "avg_chunk_size": 250,
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 95,
    "processing_time": "2024-01-01T12:00:00Z"
  }
}
```

### POST `/api/batch-chunk`
Process multiple texts in a single request.

**Headers:**
```
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

**Request:**
```json
{
  "texts": ["First text...", "Second text..."],
  "breakpoint_threshold_type": "percentile",
  "breakpoint_threshold_amount": 95
}
```

### GET `/api/health`
Health check endpoint.

## Security

The service implements enterprise-grade security features:

### Authentication
- **API Key Required**: All endpoints (except health) require Bearer token authentication
- **Multiple API Keys**: Support for multiple API keys with user identification
- **Environment Configuration**: API keys configured securely via environment variables

### Rate Limiting
- **Per-API-Key Limits**: Each API key has individual rate limits (default: 100 requests/hour)
- **Prevents Abuse**: Protects against OpenAI credit exhaustion
- **Configurable Limits**: Rate limits can be adjusted per deployment

### CORS Security
- **Restricted Origins**: CORS configured with specific allowed domains
- **Secure Headers**: Only allows necessary HTTP methods and headers
- **Environment Control**: Allowed origins configurable via environment variables

### Audit Logging
- **Request Tracking**: All API requests logged with user identification
- **Usage Monitoring**: Text length and processing metrics logged
- **Security Events**: Authentication failures and rate limit violations logged

### Configuration Example
```bash
# Security Environment Variables
API_KEYS=prod_key_123:n8n-production,dev_key_456:n8n-development
ALLOWED_ORIGINS=https://yourdomain.com,https://n8n.yourdomain.com
RATE_LIMIT_PER_KEY_HOUR=100
```

## n8n Integration

### Example Workflow

```json
{
  "nodes": [
    {
      "parameters": {},
      "id": "597e026f-617e-4f56-9bdf-bb1da744054d",
      "name": "Manual Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "typeVersion": 1,
      "position": [
        1440,
        -140
      ]
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "a0fe2c82-83dc-4887-801f-100dbac8ee43",
              "name": "text",
              "value": " Search Write Sign up  Sign in    The AI Forum The AI Forum Its AI forum where all the topics spread across Data Analytics, Data Science, Machine Learning, Deep Learning are discussed.  Follow publication  Top highlight  Semantic Chunking for RAG Plaban Nayak Plaban Nayak  Follow 17 min read · Apr 21, 2024 722   8     What is Chunking ? In order to abide by the context window of the LLM , we usually break text into smaller parts / pieces which is called chunking.  What is RAG? LLMs, although capable of generating text that is both meaningful and grammatically correct, these LLMs suffer from a problem called hallucination. Hallucination in LLMs is the concept where the LLMs confidently generate wrong answers, that is they make up wrong answers in a way that makes us believe that it is true. This has been a major problem since the introduction of the LLMs. These hallucinations lead to incorrect and factually wrong answers. Hence Retrieval Augmented Generation was introduced.  In RAG, we take a list of documents/chunks of documents and encode these textual documents into a numerical representation called vector embeddings, where a single vector embedding represents a single chunk of document and stores them in a database called vector store. The models required for encoding these chunks into embeddings are called encoding models or bi-encoders. These encoders are trained on a large corpus of data, thus making them powerful enough to encode the chunks of documents in a single vector embedding representation.  The retrieval greatly depends on how the chunks are manifested and stored in the vectorstore. Finding the right chunk size for any given text is a very hard question in general.  Improving Retrieval can be done by various retrieval method. But it can also be done by better chunking strategy.  Different chunking methods:  Fixed size chunking Recursive Chunking Document Specific Chunking Semantic Chunking Agentic Chunking Fixed Size Chunking: This is the most common and straightforward approach to chunking: we simply decide the number of tokens in our chunk and, optionally, whether there should be any overlap between them. In general, we will want to keep some overlap between chunks to make sure that the semantic context doesn’t get lost between chunks. Fixed-sized chunking will be the best path in most common cases. Compared to other forms of chunking, fixed-sized chunking is computationally cheap and simple to use since it doesn’t require the use of any NLP libraries.  Recursive Chunking : Recursive chunking divides the input text into smaller chunks in a hierarchical and iterative manner using a set of separators. If the initial attempt at splitting the text doesn’t produce chunks of the desired size or structure, the method recursively calls itself on the resulting chunks with a different separator or criterion until the desired chunk size or structure is achieved. This means that while the chunks aren’t going to be exactly the same size, they’ll still “aspire” to be of a similar size. Leverages what is good about fixed size chunk and overlap.  Document Specific Chunking: It takes into consideration the structure of the document . Instead of using a set number of characters or recursive process it creates chunks that align with the logical sections of the document like paragraphs or sub sections. By doing this it maintains the author’s organization of the content thereby keeping the text coherent. It makes the retrieved information more relevant and useful, particularly for structured documents with clearly defined sections. It can handle formats such as Markdown, Html, etc.  Sematic Chunking: Semantic Chunking considers the relationships within the text. It divides the text into meaningful, semantically complete chunks. This approach ensures the information’s integrity during retrieval, leading to a more accurate and contextually appropriate outcome. It is slower compared to the previous chunking strategy  Agentic Chunk: The hypothesis here is to process documents in a fashion that humans would do.  We start at the top of the document, treating the first part as a chunk. We continue down the document, deciding if a new sentence or piece of information belongs with the first chunk or should start a new one We keep this up until we reach the end of the document. This approach is still being tested and isn't quite ready for the big leagues due to the time it takes to process multiple LLM calls and the cost of those calls. There's no implementation available in public libraries just yet.  Here we will experiment with Semantic chunking and Recursive Retriever .  Comparison of methods steps: Load the Document Chunk the Document using the following two methods: Semantic chunking and Recursive Retriever . Assess qualitative and quantitative improvements with RAGAS Semantic Chunks Semantic chunking involves taking the embeddings of every sentence in the document, comparing the similarity of all sentences with each other, and then grouping sentences with the most similar embeddings together.  By focusing on the text’s meaning and context, Semantic Chunking significantly enhances the quality of retrieval. It’s a top-notch choice when maintaining the semantic integrity of the text is vital.  The hypothesis here is we can use embeddings of individual sentences to make more meaningful chunks. Basic idea is as follows :-  Split the documents into sentences based on separators(.,?,!) Index each sentence based on position. Group: Choose how many sentences to be on either side. Add a buffer of sentences on either side of our selected sentence. Calculate distance between group of sentences. Merge groups based on similarity i.e. keep similar sentences together. Split the sentences that are not similar. Technology Stack Used Langchain :LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications. LLM: Groq’s Language Processing Unit (LPU) is a cutting-edge technology designed to significantly enhance AI computing performance, especially for Large Language Models (LLMs). The primary goal of the Groq LPU system is to provide real-time, low-latency experiences with exceptional inference performance. Embedding Model: FastEmbed is a lightweight, fast, Python library built for embedding generation. Evaluation: Ragas offers metrics tailored for evaluating each component of your RAG pipeline in isolation. Code Implementation Install the required dependencies  !pip install -qU langchain_experimental langchain_openai langchain_community langchain ragas chromadb langchain-groq fastembed pypdf openai langchain==0.1.16  langchain-community==0.0.34  langchain-core==0.1.45  langchain-experimental==0.0.57  langchain-groq==0.1.2  langchain-openai==0.1.3  langchain-text-splitters==0.0.1  langcodes==3.3.0  langsmith==0.1.49  chromadb==0.4.24  ragas==0.1.7  fastembed==0.2.6 Download data  ! wget \"https://arxiv.org/pdf/1810.04805.pdf\" Process the PDF Content  from langchain.document_loaders import PyPDFLoader from langchain.text_splitter import RecursiveCharacterTextSplitter # loader = PyPDFLoader(\"1810.04805.pdf\") documents = loader.load() # print(len(documents)) Perform Native Chunking(RecursiveCharacterTextSplitting)  from langchain.text_splitter import RecursiveCharacterTextSplitter  text_splitter = RecursiveCharacterTextSplitter(     chunk_size=1000,     chunk_overlap=0,     length_function=len,     is_separator_regex=False ) # naive_chunks = text_splitter.split_documents(documents) for chunk in naive_chunks[10:15]:   print(chunk.page_content+ \"\\n\")  ###########################RESPONSE############################### BERT BERT  E[CLS] E1 E[SEP] ... ENE1’... EM’ C T1 T[SEP] ...  TN T1’...  TM’ [CLS] Tok 1 [SEP] ... Tok NTok 1 ... TokM  Question Paragraph Start/End Span  BERT  E[CLS] E1 E[SEP] ... ENE1’... EM’ C T1 T[SEP] ...  TN T1’...  TM’ [CLS] Tok 1 [SEP] ... Tok NTok 1 ... TokM  Masked Sentence A Masked Sentence B  Pre-training Fine-Tuning NSP Mask LM Mask LM  Unlabeled Sentence A and B Pair SQuAD  Question Answer Pair NER MNLI Figure 1: Overall pre-training and ﬁne-tuning procedures for BERT. Apart from output layers, the same architec- tures are used in both pre-training and ﬁne-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During ﬁne-tuning, all parameters are ﬁne-tuned. [CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token (e.g. separating ques- tions/answers). ing and auto-encoder objectives have been used for pre-training such models (Howard and Ruder,  2018; Radford et al., 2018; Dai and Le, 2015). 2.3 Transfer Learning from Supervised Data There has also been work showing effective trans- fer from supervised tasks with large datasets, such as natural language inference (Conneau et al., 2017) and machine translation (McCann et al., 2017). Computer vision research has also demon- strated the importance of transfer learning from large pre-trained models, where an effective recipe is to ﬁne-tune models pre-trained with Ima- geNet (Deng et al., 2009; Yosinski et al., 2014). 3 BERT We introduce BERT and its detailed implementa- tion in this section. There are two steps in our framework: pre-training and ﬁne-tuning . Dur- ing pre-training, the model is trained on unlabeled data over different pre-training tasks. For ﬁne- tuning, the BERT model is ﬁrst initialized with the pre-trained parameters, and all of the param- eters are ﬁne-tuned using labeled data from the downstream tasks. Each downstream task has sep-  arate ﬁne-tuned models, even though they are ini- tialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section. A distinctive feature of BERT is its uniﬁed ar- chitecture across different tasks. There is mini-mal difference between the pre-trained architec- ture and the ﬁnal downstream architecture. Model Architecture BERT’s model architec- ture is a multi-layer bidirectional Transformer en- coder based on the original implementation de- scribed in Vaswani et al. (2017) and released in thetensor2tensor library.1Because the use of Transformers has become common and our im- plementation is almost identical to the original, we will omit an exhaustive background descrip- tion of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as “The Annotated Transformer.”2 In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as  H, and the number of self-attention heads as A.3 We primarily report results on two model sizes: BERT BASE (L=12, H=768, A=12, Total Param- eters=110M) and BERT LARGE (L=24, H=1024, A=16, Total Parameters=340M). BERT BASE was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Trans- former uses constrained self-attention where every token can only attend to context to its left.4 1https://github.com/tensorﬂow/tensor2tensor 2http://nlp.seas.harvard.edu/2018/04/03/attention.html 3In all cases we set the feed-forward/ﬁlter size to be 4H, i.e., 3072 for the H= 768 and 4096 for the H= 1024 . 4We note that in the literature the bidirectional Trans-  Input/Output Representations To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g.,⟨Question, Answer⟩) in one token sequence. Throughout this work, a “sentence” can be an arbi- trary span of contiguous text, rather than an actual linguistic sentence. A “sequence” refers to the in- put token sequence to BERT, which may be a sin- gle sentence or two sentences packed together. We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. The ﬁrst token of every sequence is always a special clas- siﬁcation token ( [CLS] ). The ﬁnal hidden state corresponding to this token is used as the ag- gregate sequence representation for classiﬁcation tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ( [SEP] ). Second, we add a learned embed- Instantiate Embedding Model  from langchain_community.embeddings.fastembed import FastEmbedEmbeddings embed_model = FastEmbedEmbeddings(model_name=\"BAAI/bge-base-en-v1.5\") Setup the API Key for LLM  from google.colab import userdata from groq import Groq from langchain_groq import ChatGroq # groq_api_key = userdata.get(\"GROQ_API_KEY\") Perform Semantic Chunking  We’re going to be using the `percentile` threshold as an example today — but there’s three different strategies you could use on Semantic Chunking):  - `percentile` (default) — In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.  - `standard_deviation` — In this method, any difference greater than X standard deviations is split.  - `interquartile` — In this method, the interquartile distance is used to split chunks.  NOTE: This method is currently experimental and is not in a stable final form — expect updates and improvements in the coming months  from langchain_experimental.text_splitter import SemanticChunker from langchain_openai.embeddings import OpenAIEmbeddings  semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type=\"percentile\") # semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents]) # for semantic_chunk in semantic_chunks:   if \"Effect of Pre-training Tasks\" in semantic_chunk.page_content:     print(semantic_chunk.page_content)     print(len(semantic_chunk.page_content))  #############################RESPONSE############################### Dev Set Tasks MNLI-m QNLI MRPC SST-2 SQuAD (Acc) (Acc) (Acc) (Acc) (F1) BERT BASE 84.4 88.4 86.7 92.7 88.5 No NSP 83.9 84.9 86.5 92.6 87.9 LTR & No NSP 82.1 84.3 77.5 92.1 77.8 + BiLSTM 82.1 84.1 75.7 91.6 84.9 Table 5: Ablation over the pre-training tasks using the BERT BASE architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a ran- domly initialized BiLSTM on top of the “LTR + No NSP” model during ﬁne-tuning. ablation studies can be found in Appendix C. 5.1 Effect of Pre-training Tasks We demonstrate the importance of the deep bidi- rectionality of BERT by evaluating two pre- training objectives using exactly the same pre- training data, ﬁne-tuning scheme, and hyperpa- rameters as BERT BASE : No NSP : A bidirectional model which is trained using the “masked LM” (MLM) but without the “next sentence prediction” (NSP) task. LTR & No NSP : A left-context-only model which is trained using a standard Left-to-Right (LTR) LM,   Instantiate the Vectorstore  from langchain_community.vectorstores import Chroma semantic_chunk_vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model) We will “limit” our semantic retriever to k = 1 to demonstrate the power of the semantic chunking strategy while maintaining similar token counts between the semantic and naive retrieved context.  Instantiate Retrieval Step  semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={\"k\" : 1}) semantic_chunk_retriever.invoke(\"Describe the Feature-based Approach with BERT?\")  ########################RESPONSE################################### [Document(page_content='The right part of the paper represents the\\nDev set results. For the feature-based approach,\\nwe concatenate the last 4 layers of BERT as the\\nfeatures, which was shown to be the best approach\\nin Section 5.3. From the table it can be seen that ﬁne-tuning is\\nsurprisingly robust to different masking strategies. However, as expected, using only the M ASK strat-\\negy was problematic when applying the feature-\\nbased approach to NER. Interestingly, using only\\nthe R NDstrategy performs much worse than our\\nstrategy as well.')] Instantiate Augmentation Step(for content Augmentation)  from langchain_core.prompts import ChatPromptTemplate  rag_template = \"\"\"\\ Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.  User's Query: {question}  Context: {context} \"\"\"  rag_prompt = ChatPromptTemplate.from_template(rag_template) Instantiate the Generation Step  chat_model = ChatGroq(temperature=0,                       model_name=\"mixtral-8x7b-32768\",                       api_key=userdata.get(\"GROQ_API_KEY\"),) Creating a RAG Pipeline Utilizing Semantic Chunking from langchain_core.runnables import RunnablePassthrough from langchain_core.output_parsers import StrOutputParser  semantic_rag_chain = (     {\"context\" : semantic_chunk_retriever, \"question\" : RunnablePassthrough()}     | rag_prompt     | chat_model     | StrOutputParser() ) Ask Question 1  semantic_rag_chain.invoke(\"Describe the Feature-based Approach with BERT?\")  ################ RESPONSE ################################### The feature-based approach with BERT, as mentioned in the context, involves using BERT as a feature extractor for a downstream natural language processing task, specifically Named Entity Recognition (NER) in this case.  To use BERT in a feature-based approach, the last 4 layers of BERT are concatenated to serve as the features for the task. This was found to be the most effective approach in Section 5.3 of the paper.  The context also mentions that fine-tuning BERT is surprisingly robust to different masking strategies. However, when using the feature-based approach for NER, using only the MASK strategy was problematic. Additionally, using only the RND strategy performed much worse than the proposed strategy.  In summary, the feature-based approach with BERT involves using the last 4 layers of BERT as features for a downstream NLP task, and fine-tuning these features for the specific task. The approach was found to be robust to different masking strategies, but using only certain strategies was problematic for NER. Ask Question 2  semantic_rag_chain.invoke(\"What is SQuADv2.0?\") ################ RESPONSE ################################### SQuAD v2.0, or Squad Two Point Zero, is a version of the Stanford Question Answering Dataset (SQuAD) that extends the problem definition of SQuAD 1.1 by allowing for the possibility that no short answer exists in the provided paragraph. This makes the problem more realistic, as not all questions have a straightforward answer within the provided text. The SQuAD 2.0 task uses a simple approach to extend the SQuAD 1.1 BERT model for this task, by treating questions that do not have an answer as having an answer span with start and end at the [CLS] token, and comparing the score of the no-answer span to the score of the best non-null span for prediction. The document also mentions that the BERT ensemble, which is a combination of 7 different systems using different pre-training checkpoints and fine-tuning seeds, outperforms all existing systems by a wide margin in SQuAD 2.0, even when excluding entries that use BERT as one of their components. Ask Question 3  semantic_rag_chain.invoke(\"What is the purpose of Ablation Studies?\") ################ RESPONSE ################################### Ablation studies are used to understand the impact of different components or settings of a machine learning model on its performance. In the provided context, ablation studies are used to answer questions about the effect of the number of training steps and masking procedures on the performance of the BERT model. By comparing the performance of the model under different conditions, researchers can gain insights into the importance of these components or settings and how they contribute to the overall performance of the model. Implement a RAG pipeline using Naive Chunking Strategy naive_chunk_vectorstore = Chroma.from_documents(naive_chunks, embedding=embed_model) naive_chunk_retriever = naive_chunk_vectorstore.as_retriever(search_kwargs={\"k\" : 5}) naive_rag_chain = (     {\"context\" : naive_chunk_retriever, \"question\" : RunnablePassthrough()}     | rag_prompt     | chat_model     | StrOutputParser() ) Note : Here we are going to use k = 5 ;this is to “make it a fair comparison” between the two strategies.  Ask Question 1  naive_rag_chain.invoke(\"Describe the Feature-based Approach with BERT?\")  #############################RESPONSE########################## The Feature-based Approach with BERT involves extracting fixed features from the pre-trained BERT model, as opposed to the fine-tuning approach where all parameters are jointly fine-tuned on a downstream task. The feature-based approach has certain advantages, such as being applicable to tasks that cannot be easily represented by a Transformer encoder architecture, and providing major computational benefits by pre-computing an expensive representation of the training data once and then running many experiments with cheaper models on top of this representation. In the context provided, the feature-based approach is compared to the fine-tuning approach on the CoNLL-2003 Named Entity Recognition (NER) task, with the feature-based approach using a case-preserving WordPiece model and including the maximal document context provided by the data. The results presented in Table 7 show the performance of both approaches on the NER task. Ask Question 2  naive_rag_chain.invoke(\"What is SQuADv2.0?\") #############################RESPONSE########################## SQuAD v2.0, or the Stanford Question Answering Dataset version 2.0, is a collection of question/answer pairs that extends the SQuAD v1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph. This makes the problem more realistic. The SQuAD v2.0 BERT model is extended from the SQuAD v1.1 model by treating questions that do not have an answer as having an answer span with start and end at the [CLS] token, and extending the probability space for the start and end answer span positions to include the position of the [CLS] token. For prediction, the score of the no-answer span is compared to the score of the best non-null span. Ask Question 3  naive_rag_chain.invoke(\"What is the purpose of Ablation Studies?\")  #############################RESPONSE########################## Ablation studies are used to evaluate the effect of different components or settings in a machine learning model. In the provided context, ablation studies are used to understand the impact of certain aspects of the BERT model, such as the number of training steps and masking procedures, on the model's performance.  For instance, one ablation study investigates the effect of the number of training steps on BERT's performance. The results show that BERT BASE achieves higher fine-tuning accuracy on MNLI when trained for 1M steps compared to 500k steps, indicating that a larger number of training steps contributes to better performance.  Another ablation study focuses on different masking procedures during pre-training. The study compares BERT's masked language model (MLM) with a left-to-right strategy. The results demonstrate that the masking strategies aim to reduce the mismatch between pre-training and fine-tuning, as the [MASK] symbol does not appear during the fine-tuning stage. The study also reports Dev set results for both MNLI and Named Entity Recognition (NER) tasks, considering fine-tuning and feature-based approaches for NER. Ragas Assessment Comparison for Semantic Chunker split documents using RecursiveCharacterTextSplitter  synthetic_data_splitter = RecursiveCharacterTextSplitter(     chunk_size=1000,     chunk_overlap=0,     length_function=len,     is_separator_regex=False ) # synthetic_data_chunks = synthetic_data_splitter.create_documents([d.page_content for d in documents]) print(len(synthetic_data_chunks)) Create the Following Datasets  Questions — synthetically generated (grogq-mixtral-8x7b-32768) Contexts — created above(Synthetic data chunks) Ground Truths — synthetically generated (grogq-mixtral-8x7b-32768) Answers — generated from our Semantic RAG Chain questions = [] ground_truths_semantic = [] contexts = [] answers = []  question_prompt = \"\"\"\\ You are a teacher preparing a test. Please create a question that can be answered by referencing the following context.  Context: {context} \"\"\"  question_prompt = ChatPromptTemplate.from_template(question_prompt)  ground_truth_prompt = \"\"\"\\ Use the following context and question to answer this question using *only* the provided context.  Question: {question}  Context: {context} \"\"\"  ground_truth_prompt = ChatPromptTemplate.from_template(ground_truth_prompt)  question_chain = question_prompt | chat_model | StrOutputParser() ground_truth_chain = ground_truth_prompt | chat_model | StrOutputParser()  for chunk in synthetic_data_chunks[10:20]:   questions.append(question_chain.invoke({\"context\" : chunk.page_content}))   contexts.append([chunk.page_content])   ground_truths_semantic.append(ground_truth_chain.invoke({\"question\" : questions[-1], \"context\" : contexts[-1]}))   answers.append(semantic_rag_chain.invoke(questions[-1])) Note: for experimental purpose we have considered only 10 samples  Format the content generated into HuggingFace Dataset Format  from datasets import load_dataset, Dataset  qagc_list = []  for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):   qagc_list.append({       \"question\" : question,       \"answer\" : answer,       \"contexts\" : context,       \"ground_truth\" : ground_truth   })  eval_dataset = Dataset.from_list(qagc_list) eval_dataset  ###########################RESPONSE########################### Dataset({     features: ['question', 'answer', 'contexts', 'ground_truth'],     num_rows: 10 }) Implement Ragas metrics and evaluate our created dataset.   from ragas.metrics import (     answer_relevancy,     faithfulness,     context_recall,     context_precision, )  # from ragas import evaluate  result = evaluate(     eval_dataset,     metrics=[         context_precision,         faithfulness,         answer_relevancy,         context_recall,     ],      llm=chat_model,      embeddings=embed_model,     raise_exceptions=False ) Here I had tried to use open source LLM using Groq. But got a rate limit error :  groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `mixtral-8x7b-32768` in organization `org_01htsyxttnebyt0av6tmfn1fy6` on tokens per minute (TPM): Limit 4500, Used 3867, Requested ~1679. Please try again in 13.940333333s. Visit https://console.groq.com/docs/rate-limits for more information.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}  So redirected the LLM to use OpenAI which is by default used in RAGAS framework.  Set up OpenAI API keys  import os from google.colab import userdata import openai os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY') openai.api_key = os.environ['OPENAI_API_KEY'] from ragas import evaluate  result = evaluate(     eval_dataset,     metrics=[         context_precision,         faithfulness,         answer_relevancy,         context_recall,     ], ) result  #########################RESPONSE########################## {'context_precision': 1.0000, 'faithfulness': 0.8857, 'answer_relevancy': 0.9172, 'context_recall': 1.0000} #Extract the details into a dataframe results_df = result.to_pandas() results_df  Ragas Assessment Comparison for Naive Chunker import tqdm questions = [] ground_truths_semantic = [] contexts = [] answers = [] for chunk in tqdm.tqdm(synthetic_data_chunks[10:20]):   questions.append(question_chain.invoke({\"context\" : chunk.page_content}))   contexts.append([chunk.page_content])   ground_truths_semantic.append(ground_truth_chain.invoke({\"question\" : questions[-1], \"context\" : contexts[-1]}))   answers.append(naive_rag_chain.invoke(questions[-1])) Formulate naive chunking evaluation dataset  qagc_list = []  for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):   qagc_list.append({       \"question\" : question,       \"answer\" : answer,       \"contexts\" : context,       \"ground_truth\" : ground_truth   })  naive_eval_dataset = Dataset.from_list(qagc_list) naive_eval_dataset  ############################RESPONSE######################## Dataset({     features: ['question', 'answer', 'contexts', 'ground_truth'],     num_rows: 10 }) Evaluate our created dataset using RAGAS framework  naive_result = evaluate(     naive_eval_dataset,     metrics=[         context_precision,         faithfulness,         answer_relevancy,         context_recall,     ], ) # naive_result ############################RESPONSE####################### {'context_precision': 1.0000, 'faithfulness': 0.9500, 'answer_relevancy': 0.9182, 'context_recall': 1.0000} naive_results_df = naive_result.to_pandas() naive_results_df  ###############################RESPONSE ####################### {'context_precision': 1.0000, 'faithfulness': 0.9500, 'answer_relevancy': 0.9182, 'context_recall': 1.0000}  Conclusion Here we can see that the results of both Semantic Chunking and Naive Chunking are almost same except that Naive Chunker has a better factual representation of answer with a score of 0.95 when compared to score of 0.88 of Semantic Chunker.  In conclusion, semantic chunking enables the grouping of contextually similar information, allowing for the creation of independent and meaningful segments. This approach enhances the efficiency and effectiveness of large language models by providing them with focused inputs, ultimately improving their ability to comprehend and process natural language data.  References: GroqCloud Experience the fastest inference in the world console.groq.com  Metrics | Ragas Skip to content Just like in any machine learning system, the performance of individual components within the LLM and… docs.ragas.io  Semantic Chunking | 🦜️🔗 LangChain Splits the text based on semantic similarity. python.langchain.com  Semantic Chunking Langchain 722   8   The AI Forum Published in The AI Forum 984 followers · Last published Jun 8, 2025 Its AI forum where all the topics spread across Data Analytics, Data Science, Machine Learning, Deep Learning are discussed.   Follow Plaban Nayak Written by Plaban Nayak 5K followers · 394 following Machine Learning and Deep Learning enthusiast   Follow Responses (8)  Write a response  What are your thoughts?  Cancel Respond Falah Gatea Falah Gatea  Apr 25, 2024   why the RAGAS framework does not support the Gemini pro model? 2   2 replies  Reply  Lim Ming Jun Lim Ming Jun  May 7, 2024   Can you provide the source code? Like a link to the GitHub page or Google Colab? 1   1 reply  Reply  Vidyasagar Mundroy Vidyasagar Mundroy  Nov 5, 2024   At the end of the first paragraph of section \"What is RAG?\" you say hallucination of LLMs is the reason for introducing RAG. Is not getting answers to questions on \"private data\" the basic reason for introducing RAG? As I understand even RAG apps also hallucinate. Could you clarify? Reply  See all responses More from Plaban Nayak and The AI Forum 🚀 Building an AI-Powered Study Assistant with MCP, CrewAI, and Streamlit 🔬 The AI Forum In  The AI Forum  by  Plaban Nayak  🚀 Building an AI-Powered Study Assistant with MCP, CrewAI, and Streamlit 🔬 How to create a modern research companion that combines web search, AI agents, and image generation through the Model Context Protocol Jun 8 292 4 🔄 Understanding the MCP Workflow: Building a Local MCP client using Ollama and LangChain MCP… The AI Forum In  The AI Forum  by  Plaban Nayak  🔄 Understanding the MCP Workflow: Building a Local MCP client using Ollama and LangChain MCP… 🌟 What is MCP? May 24 186 1 Which Vector Database Should You Use? Choosing the Best One for Your Needs The AI Forum In  The AI Forum  by  Plaban Nayak  Which Vector Database Should You Use? Choosing the Best One for Your Needs Introduction Apr 19, 2024 715 10 🤖 Building a Multi-Agent Travel Planning System with Agent2Agent Protocol Plaban Nayak Plaban Nayak  🤖 Building a Multi-Agent Travel Planning System with Agent2Agent Protocol 🎯 Introduction Jun 28 177 1 See all from Plaban Nayak See all from The AI Forum Recommended from Medium Data Chunking Strategies for RAG in 2025 AImpact — All Things AI In  AImpact — All Things AI  by  Sachin Khandewal  Data Chunking Strategies for RAG in 2025 Exploring the latest available methods and tools to chunk data for RAG in 2025  Apr 29 91 Why Semantic Parsing is So Painful — My GraphRAG Journey Ngoc Ngoc  Why Semantic Parsing is So Painful — My GraphRAG Journey GraphRAG (Graph Retrieval-Augmented Generation) holds a lot of promise, but making it work in practice is painful — from building the…  5d ago 104 3 RAG Series 03 — Chunking Strategies in Rag Yashwanth S Yashwanth S  RAG Series 03 — Chunking Strategies in Rag Large Language Models (LLMs) like GPT-4, Claude, and LLaMA are great at generating human-like responses — but they come with context…  May 11 Re-Ranking Algorithms in Vector Databases: An In-Depth Analysis Bishal Bose Bishal Bose  Re-Ranking Algorithms in Vector Databases: An In-Depth Analysis This document provides detailed notes on five key re-ranking algorithms used in vector databases: Cross-Encoder, ColBERT, BM25, Learning to… May 24 722 Different Types of AI Agent Patterns Fundamentals of Artificial Intelligence In  Fundamentals of Artificial Intelligence  by  Arts2Survive  Different Types of AI Agent Patterns Divide responsibility across independent and focused agents  Jul 2 128 Semantic Chunking of Data for LLMs through Clustering and Classification Ludovico Attuoni Ludovico Attuoni  Semantic Chunking of Data for LLMs through Clustering and Classification In Retrieval-Augmented Generation (RAG) systems and various implementations of Large Language Models (LLMs), the challenge of effectively… Feb 14 See more recommendations Help  Status  About  Careers  Press  Blog  Privacy  Rules  Terms  Text to speech",
              "type": "string"
            },
            {
              "id": "836a0631-4ebd-414c-9d35-3809274ccdbb",
              "name": "breakpoint_threshold_type",
              "value": "percentile",
              "type": "string"
            },
            {
              "id": "01bd52c8-d94e-4440-81bd-9de26c0fedaa",
              "name": "breakpoint_threshold_amount",
              "value": "80",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "id": "fd4fd6b1-f4a0-4a86-8f96-7992bed8ee4b",
      "name": "Set Test Data",
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [
        1640,
        -140
      ]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "https://semantic-chunking-service.onrender.com/api/chunk",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ { \"text\": $json.text, \"breakpoint_threshold_type\": $json.breakpoint_threshold_type, \"breakpoint_threshold_amount\": Number($json.breakpoint_threshold_amount) } }}",
        "options": {
          "timeout": 30000
        }
      },
      "id": "aadc4d88-6d78-4874-b65e-c20a0ae3f733",
      "name": "Call Chunking API",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [
        1840,
        -140
      ],
      "credentials": {
        "httpBearerAuth": {
          "id": "RFuUauXPWescq7RB",
          "name": "Bearer Auth account"
        },
        "httpHeaderAuth": {
          "id": "ndo3yoIMqOeTg3tp",
          "name": "Header Auth account 2"
        }
      }
    },
    {
      "parameters": {
        "jsCode": "const response = $input.first().json;\n\n// Log the full response for debugging\nconsole.log('API Response:', JSON.stringify(response, null, 2));\n\n// Extract chunks and metadata\nconst chunks = response.chunks || [];\nconst metadata = response.metadata || {};\n\n// Create a formatted output\nreturn [{\n  json: {\n    chunks: chunks,\n    chunk_count: chunks.length,\n    metadata: metadata,\n    chunks_with_index: chunks.map((chunk, index) => ({\n      index: index,\n      text: chunk,\n      length: chunk.length\n    }))\n  }\n}];"
      },
      "id": "99aceb54-8b45-4066-95a5-23b59de9a332",
      "name": "Display Results",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [
        2040,
        -140
      ]
    },
    {
      "parameters": {
        "content": "## External Semantic Chunking Service (Test)\nhttps://github.com/leex279/dynamous-semantic-chunking-service\n",
        "height": 284,
        "width": 1596
      },
      "id": "a7eeeec8-f7cb-4988-8585-a8d2e470cbe2",
      "name": "Sticky Note10",
      "type": "n8n-nodes-base.stickyNote",
      "position": [
        1320,
        -240
      ],
      "typeVersion": 1
    }
  ],
  "connections": {
    "Manual Trigger": {
      "main": [
        [
          {
            "node": "Set Test Data",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Set Test Data": {
      "main": [
        [
          {
            "node": "Call Chunking API",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Call Chunking API": {
      "main": [
        [
          {
            "node": "Display Results",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  },
  "pinData": {},
  "meta": {
    "templateCredsSetupCompleted": true,
    "instanceId": "e42d6d82f39f22264e8343635848d36033cd88faee7b120a8cabc9a91605fef5"
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `API_KEYS` | API keys in format `key1:user1,key2:user2` | Required |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins | * |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute (fallback) | 60 |
| `RATE_LIMIT_PER_KEY_HOUR` | Requests per API key per hour | 100 |
| `MAX_TEXT_LENGTH` | Maximum text length | 50000 |
| `BREAKPOINT_THRESHOLD_TYPE` | Chunking threshold type | percentile |
| `BREAKPOINT_THRESHOLD_AMOUNT` | Threshold amount | 95 |
| `CACHE_SIZE` | Number of cached results | 100 |
| `LOG_LEVEL` | Logging level | INFO |

## Performance

### Benchmarks
- **Startup Time**: 30-60 seconds (render.com spin-up)
- **Small texts** (<1000 chars): <2 seconds
- **Medium texts** (1000-5000 chars): 2-5 seconds
- **Large texts** (>5000 chars): 5-15 seconds

### Resource Usage
- **Memory**: 150-200MB typical usage
- **CPU**: 80-90% during processing
- **Cache Hit Rate**: 30-50% for common content

## Cost Optimization

- **OpenAI Embeddings pricing**: ~$0.00002 per 1000 tokens (text-embedding-3-small)
- **Intelligent caching**: Reduces repeated embedding calls
- **Batch processing**: Efficient for multiple texts
- **Semantic chunking**: More accurate than fixed-size chunking

## Security

- API key validation
- Rate limiting per endpoint
- Input size validation
- Secure environment variable handling

## Development

### Project Structure
```
semantic-chunking-service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── render.yaml         # Render.com deployment config
├── .env.example        # Environment variables template
├── PLANNING.md         # Implementation plan
└── README.md          # This file
```

### Local Testing
```bash
# Start the service
python main.py

# Test health endpoint
curl http://localhost:8000/api/health

# Test chunking endpoint
curl -X POST http://localhost:8000/api/chunk \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text here", "breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 95}'
```

## Monitoring

The service includes comprehensive logging and health checks:
- Request/response logging
- Error tracking
- Performance metrics
- Cache statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the PLANNING.md for implementation details
- Review the logs for troubleshooting