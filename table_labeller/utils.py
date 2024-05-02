

#open curr_preds.json
from collections import defaultdict
import json
import random
import requests

url = "http://localhost:8080/api/predictions/"
token = "782ba8a7e729c9e106bf3ae69b0c217daad7103e"

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


def request_interactive_annotation(api_url, backend_id, token, task_id, context=None):
    headers = {
        'Authorization': f'Token {token}',
        'Content-Type': 'application/json'
    }
    payload = {
        "task": task_id,
        "context": context if context is not None else {}
    }
    response = requests.post(f"{api_url}/{backend_id}/interactive-annotating", headers=headers, data=json.dumps(payload))
    return response.status_code, response.json()


def label_predictions(data, api_url, backend_id, token):
    count = 0
    for prediction in data:
        task_id = prediction['id']
        if prediction["predictions"] == []:
            try:
                status_code, response_data = request_interactive_annotation(api_url, backend_id, token, task_id)
                
                _response_data = response_data["data"]

                payload = {
                    "model_version": _response_data["model_version"],
                    "result": _response_data["result"],
                    "score": 0,
                    "cluster": 0,
                    "neighbors": { },
                    "mislabeling": 0,
                    "task": task_id,
                    "project": 12
                }
                _cstatus, response_data = create_prediction(url, token, payload)


                # randomly sleep for 1-3 seconds
                import time
                import random
                # time.sleep(random.randint(1, 3))
                count += 1
                print(f"{count}: Task ID: {task_id} - Status: {_cstatus} - Response: {response_data}")
            except Exception as e:
                print(f"Error: {e}")
                continue

def get_json_from_label_studio(url):

    status, response = requests.get(url)
    if status == 200:
        return response.json()
    
    return None


# api_url = "http://localhost:8080/api/ml"
# task_list_url = "http://localhost:8080/api/projects/12/export?exportType=JSON&download_all_tasks=true"
# predictions_list_url = "http://localhost:8080/api/predictions"

# task_data = get_json_from_label_studio(task_list_url)
# predictions_data = get_json_from_label_studio(predictions_list_url)
# count = 0
# unknown_count = 0

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

