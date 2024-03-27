from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import json
import openai
import re
import pandas as pd
from load_selfknowledge import simple_search_with_context_bert
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
openai.api_key = "sk-"
def extract_matching_sentences(text, pattern):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    matching_sentences = []
    for sentence in sentences:
        if re.search(re.escape(pattern), sentence):  # 使用re.escape确保pattern被正确处理
            matching_sentences.append(sentence)
    
    return matching_sentences 

def retrieve_documents(query, knowledge_base='beir-v1.0.0-bioasq-flat', top_n=10):
    # initialize the retrieval 
    # problem here is the knowledge is too long
    if knowledge_base == "wikipedia":
        searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
        index_reader = IndexReader.from_prebuilt_index('enwiki-paragraphs')
        # vectorizer = BM25Vectorizer('enwiki-paragraphs')
    elif knowledge_base == "pubmed":
        searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq.flat')
        index_reader = IndexReader.from_prebuilt_index('beir-v1.0.0-bioasq.flat')
        # vectorizer = BM25Vectorizer('beir-v1.0.0-bioasq.flat')
    else:
        raise ValueError("Unsupported knowledge base.")


    hits = searcher.search(query, k=top_n)

    documents = []
    for i, hit in enumerate(hits):
        docid = hit.docid
        score = hit.score
        if knowledge_base =="wikipedia":
            raw_document = searcher.doc(docid).raw()
            # raw_document = '\t'.join([raw_document for hit in hits[:1]])
        else:
            raw_document = searcher.doc(docid).raw()
        
            raw_document = json.loads(raw_document)["text"]
        # documents.append({"docid": docid, "score": score, "document": raw_document})
        # print(f"Rank: {i+1}, DocID: {docid}, Score: {score}")
        print(raw_document)
        print("-" * 80)
    # return documents
    return raw_document


def retrieval_from_selfknowledge(query):
    context_texts = simple_search_with_context_bert(query,context_range=1)
    return " ".join(context_texts) if len(context_texts)>0 else " "
def llm_figure_specious_sentence(text_description):
    response =  openai.chat.completions.create(
    # model="text-davinci-003", 
    model="gpt-3.5-turbo", # gpt-4
    messages=[
        { "role": "user",
         "content":"The EHR note 'A 24-year-old woman comes to the emergency department because of a 4-hour \
        history of headaches, nausea, and vomiting. During this time, she has also had recurrent dizziness\
        and palpitations. The symptoms started while she was at a friend's birthday party, where she had one\
        beer. One week ago, the patient was diagnosed with a genitourinary infection and started on\
        antimicrobial therapy. She has no history of major medical illness. Culture tests indicate\
        Neisseria gonorrhoeae. Her pulse is 106/min and blood pressure is 102/73 mm Hg. Physical\
        examination shows facial flushing and profuse sweating.' in this EHR note the error \
        sentence is 'Culture tests indicate Neisseria gonorrhea.' the error span is\
        'Neisseria gonorrhea' it should be 'Culture tests indicate Trichomonas vaginalis.' \
        Then I will provide another EHR note which may has error please help me find it out where \
        is the error sentence and if possible generate the right statement. Here is the EHR note '{}'. You should only generate the suspecious error sentence without any explaination if there is no suspecious error\
        please generate None".format(text_description),},
    ],
    )

    suspicious_sentence_pattern = response.choices[0].message.content
    print("Generated suspicious sentence:", suspicious_sentence_pattern)

    return suspicious_sentence_pattern



def load_local_data(trainpath, validpath):
    trainpath = "data/csnlp/MEDIQA-CORR-2024-MS-TrainingData.csv"
    validpath = "data/csnlp/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"
    
    train_df = pd.read_csv(trainpath)
    valid_df = pd.read_csv(validpath)

    train_text = train_df.Text.values.tolist()
    train_label = train_df["Error Flag"].values.tolist()
    train_wrongsen = train_df["Error Sentence"].values.tolist()

    valid_text = valid_df.Text.values.tolist()
    valid_label = valid_df["Error Flag"].values.tolist()
    valid_wrongsen = valid_df["Error Sentence"].values.tolist()
    return train_text, train_label, train_wrongsen, valid_text, valid_label, valid_wrongsen


def llm_generate_reason(full_text, suspecious, knowledge, label):
    if label == 0: # not right
        content = ("Given the background information and the EHR note: '{}', explain why the specific description: '{}' is incorrect "
           "based on the additional knowledge provided: '{}'. Focus on the discrepancies between the description and the "
           "knowledge to provide a detailed explanation. ").format(full_text, suspecious, knowledge)

    else: # is right just say it have matched result
        content = ("Given the background information and the EHR note: '{}', confirm why the specific description: '{}' is accurate "
           "based on the additional knowledge provided: '{}'. Highlight how the description aligns with the knowledge to "
           "substantiate the correctness.").format(full_text, suspecious, knowledge)

    # main_text = full_text-suspecious # main text is the place where without the information of the wrong sentence
    response = openai.chat.completions.create(
    # model="text-davinci-003", 
    model="gpt-3.5-turbo", # gpt-4
    messages=[
        { "role": "user",
        "content":content,},
        ],
    max_tokens=100,  
    temperature=0.7, 
    )
    
    reasons = response.choices[0].message.content
    print("Generated reason:", reasons)
    
    return reasons

def llama_generate_reason():
    pass

def clean_data(full_text, suspecious):
    # verify the suspecious is in the input text
    sentences = re.findall(r'\b.*?[\.\?!]', suspecious, re.DOTALL)
    
    for sentence in sentences:
        # remove the space
        trimmed_sentence = sentence.strip()
        # check where sentence in the full text
        if trimmed_sentence in full_text:
            print("trimmed_sentence", trimmed_sentence)
            return trimmed_sentence
        
    return "No matching sentence found."
 

def remove_suspious(full_text, suspious):
     # Splitting text into lists of sentences using NLTK
    sentences = sent_tokenize(full_text)
    
    # Remove the specified sentence
    sentences = [sentence for sentence in sentences if sentence != suspious]
    
    # Reconnecting the remaining sentences into new text
    new_text = ' '.join(sentences)
    
    return new_text

def check_overlap(sentence1, sentence2):
    # Split sentences into words and convert to lowercase to ignore case differences
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    
    # Find the intersection of two sets
    overlap = words1.intersection(words2)
    
    # if more than 0 means have the overlap
    return len(overlap) > 0

if __name__ == "__main__":
    # train_text, train_label, train_wrongsen, valid_text,  valid_label, valid_wrongsen= \
    #     load_local_data()

    # full_text = """A 3100-g (6.9-lb) male newborn is brought to the emergency department by his mother because of fever and irritability. The newborn was delivered at home 15 hours ago. He was born at 39 weeks' gestation. The mother's last prenatal visit was at the beginning of the first trimester. She received all standard immunizations upon immigrating from Mexico two years ago. Seven weeks ago, she experienced an episode of painful, itching genital vesicles, which resolved spontaneously. Four hours before going into labor she noticed a gush of blood-tinged fluid from her vagina. The newborn is ill-appearing and lethargic. His temperature is 39.9 C (103.8 F), pulse is 170/min, respirations are 60/min, and blood pressure is 70/45 mm Hg. His skin is mildly icteric. Expiratory grunting is heard on auscultation. Skin turgor and muscle tone are decreased. Laboratory studies show:
    # Hemoglobin 15 g/dL
    # Leukocyte count 33,800/mm3
    # Platelet count 100,000/mm3
    # Serum glucose 55 mg/dL
    # Diagnosis was caused by herpes simplex virus."""
    # pre_suspecious = llm_figure_specious_sentence(full_text)
    # suspecious = clean_data(full_text, pre_suspecious)

    # no_query = """A 3100-g (6.9-lb) male newborn is brought to the emergency department by his mother because of fever and irritability. The newborn was delivered at home 15 hours ago. He was born at 39 weeks' gestation. The mother's last prenatal visit was at the beginning of the first trimester. She received all standard immunizations upon immigrating from Mexico two years ago. Seven weeks ago, she experienced an episode of painful, itching genital vesicles, which resolved spontaneously. Four hours before going into labor she noticed a gush of blood-tinged fluid from her vagina. The newborn is ill-appearing and lethargic. His temperature is 39.9 C (103.8 F), pulse is 170/min, respirations are 60/min, and blood pressure is 70/45 mm Hg. His skin is mildly icteric. Expiratory grunting is heard on auscultation. Skin turgor and muscle tone are decreased. Laboratory studies show:
    # Hemoglobin 15 g/dL
    # Leukocyte count 33,800/mm3
    # Platelet count 100,000/mm3
    # Serum glucose 55 mg/dL"""
    # no_query = remove_suspious(full_text, suspecious)
    # # suspecious = "Diagnosis was caused by herpes simplex virus."
    # # retrieve_documents(query, knowledge_base='wikipedia', top_n=1) # Herpes simplex . It should not be confused with conditions caused by other viruses in the "herpesviridae" family such as herpes zoster, which is caused by varicella zoster virus. The differential diagnosis includes hand, foot and mouth disease due to similar lesions on the skin.
    # knowledge = retrieve_documents(no_query, knowledge_base='pubmed', top_n=10) 
    # print("knowledge: ", knowledge)
    # reason = llm_generate_reason(full_text,suspecious,knowledge, 0)
    
    import json
    trainpath = "data/csnlp/MEDIQA-CORR-2024-MS-TrainingData.csv"
    validpath = "data/csnlp/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"
    train_text, train_label, train_wrongsen, valid_text,  valid_label, valid_wrongsen= \
            load_local_data(trainpath, validpath)

    right_predict_sus =0

    csnlp = []
    # extract first 10 example
    for idx, full_text in enumerate(train_text[1500:]):
        try:
            res = {}
            pre_suspecious = llm_figure_specious_sentence(full_text)
            suspecious = clean_data(full_text, pre_suspecious)
            if train_label[idx] == 1:
                if suspecious == "No matching sentence found." or not check_overlap(suspecious, train_wrongsen[idx]):
                    suspecious = train_wrongsen[idx]
                    print("wrong prediction of the first 10 example: {} and the suspecious is: {}".format(idx, suspecious))
                    print("right suspecious is: {}".format(train_wrongsen[idx]))
                else:
                    right_predict_sus += 1
            else:
                suspecious = None

            no_query = remove_suspious(full_text, suspecious)
            knowledge = retrieve_documents(no_query, knowledge_base='pubmed', top_n=1)
            print("knowledge: ", knowledge)
            reason = llm_generate_reason(full_text, suspecious, knowledge, train_label[idx])
            res["reason"] = reason
            res["full_text"] = full_text
            res["knowledge"] = knowledge
            res["label"] = train_label[idx]

            # Save each result to file immediately before appending
            with open('data/csnlp/csnlp_train.jsonl', 'a') as f:
                json.dump(res, f)
                f.write('\n')

            csnlp.append(res)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")

        print("right_predict_sus", right_predict_sus)