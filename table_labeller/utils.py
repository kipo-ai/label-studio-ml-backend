

#open curr_preds.json
from collections import defaultdict
import json
import random
import requests
from table_labeller.model import TableModel


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def delete_prediction(prediction_id, url, token):
    headers = {
        'Authorization': f'Token {token}'
    }
    response = requests.delete(f"{url}{prediction_id}", headers=headers)
    return response.status_code, response.json()


def delete_all_predictions(data, url, token):
    for prediction in data:
        try:
            status_code, response_data = delete_prediction(prediction['id'], url, token)
            print(status_code, response_data)
        except Exception as e:
            print(f"Error: {e}")
            continue


def create_prediction(api_url, token, payload):
    headers = {
        'Authorization': f'Token {token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    return response.status_code, response.json()


def request_interactive_annotation(api_url, backend_id, token, task_id, context={}):
    headers = {
        'Authorization': f'Token {token}',
    }
    payload = {
        "task": task_id,
        "context": context if context is not None else {}
    }

    response = requests.post(f"{api_url}/{backend_id}/interactive-annotating", headers=headers, data=json.dumps(payload))
    return response.status_code, response.json()


def get_results_from_local(url, task):

    m = TableModel()    
    respose = m.predict([task])

    return respose


def label_predictions(data, api_url, backend_id, token):
    count = 0
    for task in data:
        task_id = task['id']
        if task["predictions"] == []:
            try:
                response_data = get_results_from_local(api_url, task)

                payload = {
                    "model_version": response_data[0]["model_version"],
                    "result": response_data[0]["result"][0]["value"]["choices"][0],
                    "score": 0,
                    "cluster": 0,
                    "neighbors": {},
                    "mislabeling": 0,
                    "task": task_id,
                    "project": backend_id
                }

                _cstatus, response_data = create_prediction("https://labelstudio.kipo.ai/api/predictions", token, payload)
                count += 1
                print(f"{count}: Task ID: {task_id} - Status: {_cstatus} - Response: {response_data}")
            except Exception as e:
                print(f"Error: {e}")
                continue


def get_json_from_label_studio(url, token):

    headers = {"Authorization": f"Token {token}"}
    
    # Make the GET request with headers
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    
    return None


backend_id = 23
token = "dcb2623d0c78e5be334a2e72bddc3abb918a37f9"
api_url = "https://labelstudio.kipo.ai/api/ml"
task_list_url = f"https://labelstudio.kipo.ai/api/projects/{backend_id}/export?exportType=JSON&download_all_tasks=true"
# predictions_list_url = "http://localhost:8080/api/predictions"


# if curr_tasks.json exists, use it and avoid making a request to the server
try:
    task_data = read_json("curr_tasks.json")
except:
    task_data = get_json_from_label_studio(task_list_url, token)

# save the task data to a file
with open("curr_tasks.json", "w") as f:
    json.dump(task_data, f)



# predictions_data = get_json_from_label_studio(predictions_list_url)
count = 0
unknown_count = 0

print(f"Total tasks: {len(task_data)}")

# label all tasks
label_predictions(task_data, api_url, backend_id, token)

# unique_labels_count = defaultdict(list)
# for prediction in predictions_data:

#     if prediction["result"] == [] or prediction["result"] == None:
#         unknown_count += 1
#         continue

#     try:
#         label = prediction['result'][-1]["value"]["choices"][0]
#     except:
#         breakpoint()
#         print(prediction)
#         break

#     unique_labels_count[label].append(prediction["task"])


# print(f"Total predictions: {len(predictions_data)}")
# print(f"Unknown predictions: {unknown_count}")

# for label, task_ids in unique_labels_count.items():
#     print(f"Label: {label} - Count: {len(task_ids)}")

#     # select 10 random task ids
#     rand_task_ids = random.sample(task_ids, 10)
    
#     # get the task data
#     for task_id in rand_task_ids:
#         task = [task for task in task_data if task['id'] == task_id]
#         [print(f'{task["data"]["text"]}\n') for task in task]

#     print("\n\n")





# curl -X POST 'https://labelstudio.kipo.ai/api/predictions/' \
#      -H 'Authorization: Token dcb2623d0c78e5be334a2e72bddc3abb918a37f9' \
#      -H 'Content-Type: application/json' \
#      -d '{
#            "model_version": "0.0.1", 
#            "task": 169666, 
#            "project": 22
#          }'
