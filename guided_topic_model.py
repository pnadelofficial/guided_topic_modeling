from docx import Document
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import collections
from sklearn.cluster import KMeans
from nltk import sent_tokenize
import tiktoken
import os
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import partial

QUESTION_PROMPT = """
# Applied Research Task
Given a user provided text, please create a list of distinct topics that the text describes and then for each topic compose a list of 2 to 5 research questions. Delimit topics with `###` and research questions with `\n*`.
Though the input text is very theoretical, these research questions should be as practical as possible, relating the lives of those who the text is about.
Please do not compose questions that discuss promoting equity and social justice, such as: "What are the ethical considerations surrounding the representation of diseases in the media, and how can we ensure that media portrayals are accurate and respectful?" Though these questions are important, they are not relevant to this project.
They will be compared against the testimonies of those suffering from diseases, so it is important that your questions both touch on the theoretical components of the text and this lived experience.
Please compose at least 5 to 10 topics. Do not repeat any questions. Make sure to format your response in markdown.
"""

LABEL_PROMPT = """
# Labeling task
Based on the following sample of 20 documents and a query, please compose a single-sentence label which describes the relationship between the documents and the query. Do not mention the query or anything about it in the label. Do not provide an explanation of your label.

## Documents separated by new lines
{examples}

## Query
{query}

## Concise, descriptive label
"""

dross = [
    "This text is irrelevant.", 
    "This text does not make sense.", 
    "This text is gibberish.", 
    "This text is too short to be meaningful."
    ]

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

class GuidedTM:
    def __init__(
        self,
        model_path, 
        theory_document,
        documents_path,
        embedding_model,
        bnb_config=None,
        text_col=None,
        embeddings_path=None,
        sample_for_labels=10,
        chunk_size=250, 
        chunk_overlap=50, 
        token_chunking=True,
        system_prompt=False,
        theory_questions=None
    ):
        self.model_path = model_path
        self.theory_document = theory_document
        self.documents_path = documents_path
        self.embedding_model = embedding_model
        self.bnb_config = bnb_config
        self.theory_questions = theory_questions
        
        self.text_col = text_col
        self.embeddings_path = embeddings_path
        self.sample_for_labels = sample_for_labels
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = torch.device('cuda')  
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': self.device}, encode_kwargs={'normalize_embeddings': True})
        self.model = self.embeddings.client
        self.token_chunking = token_chunking
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) if self.token_chunking else SemanticChunker(self.embeddings)
        self.system_prompt = system_prompt

    def _get_theory_text(self):
        doc = Document(self.theory_document)
        self.theory_text =  ''.join([p.text for p in doc.paragraphs if (p.text != '') and (p.text != '  ')])

    def _init_model(self):
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto', pad_token_id=self.tokenizer.eos_token_id, quantization_config=self.bnb_config)

    def _init_messages(self):
        if self.system_prompt:
            messages = [
                {'role':'system', 'content':QUESTION_PROMPT},
                {'role':'user', 'content':self.theory_text}
            ]
        else:
            prompt = QUESTION_PROMPT + "\n## Provided Text:" + self.theory_text + "## List of topics and questions:"
            messages = [
                {'role':'user', 'content':prompt}
            ]
            
        return self.tokenizer.apply_chat_template(messages, return_tensors='pt')            
        
    def _get_questions_from_theory(self):
        text = self._init_messages().to(self.device)
        generated = self.llm.generate(text, max_new_tokens=3000, do_sample=True)
        self.topics_and_questions = self.tokenizer.batch_decode(generated)[0].split("[/INST]")[1]
        topic_list = re.split(r'(?=\#\#\#)', self.topics_and_questions)[1:]
        topic_list = [re.split(r'\n\*',t) for t in topic_list]
        self.topic_dict = {t[0].strip():[q.strip().replace('</s>', '') for q in t[1:]] for t in topic_list}
        self.topic_dict['Irrelevant'] = dross

    def _chunk_text(self, text):
        if self.token_chunking:
            return self.text_splitter.split_text(text)
        else:
            docs = self.text_splitter.create_documents([text])
            return [d.page_content for d in docs]

    def _embed(self, fn_save=f"embeddings_{date.today()}.npz"):
        self.df = pd.read_csv(self.documents_path)
        self.df = self.df[~self.df.text.str.isspace()]
        self.chunks = self.df[self.text_col].dropna().apply(self._chunk_text).explode()
        self.embeddings = self.model.encode(self.chunks.to_list(), device=self.device, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        if fn_save:
            np.savez_compressed(fn_save, self.embeddings)
    
    def _init_docs(self):
        if self.embeddings_path:
            loaded_data = np.load(self.embeddings_path)
            self.embeddings = loaded_data['arr_0']
            self.model = SentenceTransformer(self.embedding_model)

            self.df = pd.read_csv(self.documents_path)
            self.chunks = self.df[self.text_col].dropna().apply(self._chunk_text).explode().to_list()
        else:
            self._embed(self.text_col)

    def _form_centroids(self):
        embedding_dict = collections.defaultdict(list)

        for key,value in self.topic_dict.items():
            for question in value:
                query_embedding = self.model.encode(QUERY_INSTRUCTION+question.strip(), device=self.device, normalize_embeddings=True)
                top_k = (self.embeddings @ query_embedding).argsort()[::-1][0]
                top_k_embeddings = self.embeddings[top_k]
                embedding_dict[key].append(top_k_embeddings)

        embedding_dict = dict(embedding_dict)
        centroids = [[a for a in v] for k,v in embedding_dict.items()]
        centroids = np.array([x for xs in centroids for x in xs])
        self.centroids = centroids

    def _init_kmeans(self):
        num_clusters = self.centroids.shape[0]
        self.kmeans = KMeans(n_clusters=num_clusters, init=self.centroids, n_init=1)

    def _cluster_around_centroids(self):
        self.kmeans.fit(self.embeddings)
        cluster_assignments = self.kmeans.predict(self.embeddings)
        chunked_text = pd.Series(self.chunks, name=self.text_col).reset_index()
        chunked_text['label'] = cluster_assignments
        chunked_text = chunked_text.drop('index', axis=1)
        self.chunked_text = chunked_text

    def _format_and_batch_label_prompts(self):
        self.topics = self.chunked_text.copy()
        self.questions = [x.strip() for xs in [v for _,v in self.topic_dict.items()] for x in xs]
        topic_idxs = sorted(self.chunked_text.label.unique())

        nums_tokens = []
        self.prompts_as_messages = []
        self.not_labeled = [] 
        for i, question in zip(topic_idxs, self.questions):
            examples_series = self.chunked_text[self.chunked_text['label'] == i]
            if len(examples_series) < self.sample_for_labels:
                print(f"Label {i}: Not enough documents to label")
                self.not_labeled.append(i)
                continue
            else:
                examples = "\n".join(examples_series.sample(self.sample_for_labels)[self.text_col].to_list())
                tokenized = self.tokenizer.encode(examples)
                nums_tokens.append(len(tokenized))

                filled_label_prompt = LABEL_PROMPT.format(examples=examples, query=question)
                messages = [
                    {'role':'user', 'content':filled_label_prompt}
                ]
                text = self.tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
                self.prompts_as_messages.append(text)

        self.tokenized_messages = self.tokenizer(self.prompts_as_messages, return_tensors="pt", padding=True).to(self.device)
        self.max_size = max(nums_tokens)

    def _generate_labels(self):
        self.labels = []
        generated_ids = self.llm.generate(**self.tokenized_messages, max_new_tokens=250)
        self.labels.append(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        self.labels = [x for xs in self.labels for x in xs]

    def _format_label_dict(self):
        labels_and_questions = [(q.split('\n\n')[0],l.split('[/INST]')[1].strip().split(':')[-1].strip()) for q,l in zip(self.questions, self.labels)]
        labeled = list(set([i for i in range(len(self.questions))]) - set(self.not_labeled))
        labeled_mapping = dict(zip(labeled, [i for i in range(len(labels_and_questions))]))

        label_dict = collections.defaultdict(str)
        for i in range(len(self.topics.label.value_counts())):
            if i in labeled:
                label_dict[i] = labels_and_questions[labeled_mapping[i]]
            else:
                label_dict[i] = 'not labeled'
        self.label_dict = label_dict

    def _format_df(self):
        self.topics['label_string'] = self.topics.label.apply(lambda x: label_dict[x][0])
        self.topics['question_string'] = self.topics.label.apply(lambda x: label_dict[x][1])

    def __call__(self):
        if not self.theory_questions:
            self._get_theory_text()
        self._init_model()
        self._init_messages()
        self._get_questions_from_theory()
        self._init_docs()
        self._form_centroids()
        self._init_kmeans()
        self._cluster_around_centroids()
        self._format_and_batch_label_prompts()
        self._generate_labels()
        self._format_label_dict()
        self._format_df
        
        return self.topics
    
