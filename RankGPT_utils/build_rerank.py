import re
from tqdm import tqdm
from rank_gpt_reason import sliding_windows, create_permutation_instruction, run_llm, receive_permutation 



def reason_search(json_file):
    return_format = []
    for item in tqdm(json_file):
        query = item["query"]
        # (1) Create permutation generation instruction
        messages = create_permutation_instruction(item=item, rank_start=0, rank_end=5, model_name='gpt-3.5-turbo')
        # (2) Get ChatGPT predicted permutation
        permutation = run_llm(messages, api_key="sk-", model_name='gpt-3.5-turbo')
        # (3) Use permutation to re-rank the passage
        reasoning = permutation
        # print("permutation",permutation)
        # Extract the numbers in the part
        def clean_reason(permutation):
            
            ranking_pattern = re.findall(r'\[\d+\] >', permutation)
            new_response = ""
            
            ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
            new_response = " ".join([str(i )for i in ranking_numbers])
           
            last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', permutation)
            if last_number_match:
                last_number = last_number_match.group().strip('[]')
                new_response +=  " " + last_number
                # ranking_numbers.append(str(last_number))
            # print("new_response",new_response)
            return new_response
        res = clean_reason(permutation)
        response_id = []
        res = res.split()
        
        for i in res:
            response_id.append(item["hits"][int(i)-1]["docid"])
        print("response_id", response_id)
        return_format.append({"query":query, "sorted_docids":response_id, "reason": reasoning })
    return return_format


def slide_window()
api_key = "Your OPENAI Key"
new_item = sliding_windows(item, rank_start=0, rank_end=3, window_size=2, step=1, model_name='gpt-3.5-turbo', api_key=api_key)
print(new_item)

print("bm25 reason search")
reason_bm25 = reason_search(refined_bm25_file)
print("random reason search")
reason_random = reason_search(refined_random_file)