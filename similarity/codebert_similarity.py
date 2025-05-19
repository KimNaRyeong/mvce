import json, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaModel.from_pretrained(model_name).to(device)
model.eval()

def calculating_similarity(embedding_dict):
    average_similarity_dict = {}

    print("Calculating cosine similarity...")
    for prob, embeddings in tqdm(embedding_dict.items()):
        embedding_tensor = torch.stack(embeddings).to(device)
        norm_embeddings = F.normalize(embedding_tensor, p=2, dim=1)

        sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)

        num_answers = sim_matrix.size(0)
        mask = torch.eye(num_answers, dtype = torch.bool, device = sim_matrix.device)
        sim_matrix.masked_fill_(mask, 0.0)

        avg_similarities = sim_matrix.sum(dim=1) / (num_answers - 1)

        average_similarity_dict[prob] = avg_similarities.cpu().tolist()
    
    return average_similarity_dict

def generating_embedding(results, benchmark):
    embedding_dict = {}

    print("Generating embeddings...")
    for prob, answers in tqdm(results.items()):
        embedding_dict[prob] = []
        if benchmark == 'HumanEval':
            for a in answers:
                answer_body = '\n'.join(
                    [
                        line for line in a[0].split('\n')
                        if not line.startswith('```') and not line.strip().startswith('def')
                    ]
                )

                tokens = tokenizer(
                    answer_body,
                    return_tensors = 'pt',
                    max_length = 512,
                    truncation = True,
                    padding = 'max_length'
                ).to(device)

                with torch.no_grad():
                    outputs = model(**tokens)
                    last_hidden_state = outputs.last_hidden_state

                    cls_embedding =  last_hidden_state[:, 0, :]
                    cls_embedding =  cls_embedding.squeeze(0).cpu()
                    embedding_dict[prob].append(cls_embedding)
    
    return embedding_dict


def get_final_answer(results_path, benchmark):
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    embedding_dict = generating_embedding(results, benchmark)
    similarity_dict = calculating_similarity(embedding_dict)

    final_answer_dict = {}
    
    for prob, similarities in similarity_dict.items():
        final_answer_dict[prob] = {}

        max_value = max(similarities)
        max_index = similarities.index(max_value)

        final_answer = results[prob][max_index]
        final_answer_body = '\n'.join(
                                [
                                    line for line in final_answer[0].split('\n')
                                    if not line.startswith('```') and not line.strip().startswith('def')
                                ]
                            )
        final_answer_result = final_answer[1]["result"]

        final_answer_dict[prob]["similarity"] = max_value
        final_answer_dict[prob]["body"] = final_answer_body
        final_answer_dict[prob]["result"] = final_answer_result
        final_answer_dict[prob]["similarities"] = similarities
    
    return final_answer_dict

def evaluate(final_answer_dict):
    num_passed = 0
    num_total = len(final_answer_dict.keys())
    for final_answer in final_answer_dict.values():
        if final_answer["result"] == "passed":
            num_passed += 1
    
    return num_passed, num_total
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', default='HumanEval', type=str, help='Benchmark name')
    parser.add_argument('-r', '--responses_path', default='../results/HumanEval_llama3.json')
    args = parser.parse_args()
    assert args.benchmark in['HumanEval', 'APPS']

    final_answer_dict = get_final_answer(args.responses_path, args.benchmark)
    num_passed, num_total = evaluate(final_answer_dict)
    print(f"num_passed: {num_passed}")
    print(f"num_total: {num_total}")

    result_file = args.responses_path.split('/')[-1]
    result_path = f"./{result_file}"

    with open(result_path, 'w') as f:
        json.dump(final_answer_dict, f, indent = 4)

    
