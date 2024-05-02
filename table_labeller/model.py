from enum import Enum
import json
from math import exp
import os
from typing import Any, List, Dict, Optional, Tuple, Union
import uuid

from dotenv import load_dotenv
from constants.LLMConstants import GPT3
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from utils.openai_utils import json_check_wrapped_call_gpt


class TableCategory(Enum):
    TABLE_OF_CONTENTS = "table_of_contents"
    PIN_TABLE = "pin_table"
    PIN_MAP = "pin_map"
    SPECIFICATIONS = "specs_table"
    PACKAGE_INFORMATION = "package_information_table"
    NOT_A_TABLE = "not_a_valid_table"
    UNKNOWN = "unknown"
    DIMENSIONAL = "dimensional_table"

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")


    def create_classification_prompt(self, markdown_table: str) -> str:
        """
        Generates a prompt for classifying the table from technical datasheets.
        """
        return (
            "Please classify the following table from a technical datasheet of electronic components. "
            "Each table can be categorized based on specific characteristics and content. The categories are:\n\n"
            "- table_of_contents: Lists the sections and their respective page numbers; typically non-technical.\n"
            "- pin_table: Contains details such as pin numbers, functions, and descriptions; crucial for hardware design.\n"
            "- pin_map: Provides a graphical or schematic representation of pin layouts; usually accompanied by diagrams.\n"
            "- specs_table: Outlines technical specifications like voltage, current, dimensions, and operating parameters.\n"
            "- package_information_table: Includes details about the physical packaging of the component like size, pin count, and form factor.\n"
            "- not_a_valid_table: The content does not represent a structured table; may include random text or images.\n"
            "- dimensional_table: Contains measurements, dimensions, and tolerances; essential for mechanical design.\n"
            "- unknown: The content of the table is unclear, unspecified, or not sufficiently detailed to categorize accurately.\n\n"
            "Here is the markdown content of a table:\n"
            f"{markdown_table}\n\n"
            "Based on the content, which category best describes this table? Provide the category name."
            "Provide the answer in the following format:\n"
            "json\n"
            "{\n"
            '  "category": "table_of_contents"\n'
            "}\n"
            "Acceptable values for 'category' are: table_of_contents, pin_table, pin_map, specs_table, package_information_table, not_a_valid_table, dimensional_table, unknown."
        )



    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}
        # Extra params: {self.extra_params}''')

        markdown_table = tasks[0]["data"]["text"]
        system_prompt = (
            "You are an AI Labeller with expert knowledge about electronics. "
            "You follow instructions carefully and precisely. "
        )

        prompt = self.create_classification_prompt(markdown_table)

        for _ in range(3):
            response = json_check_wrapped_call_gpt(
                system_message=system_prompt,
                user_message_content=prompt,
                model=GPT3,
            )

            if response["category"] not in [cat.value for cat in TableCategory]:
                continue
            else:
                category = response["category"]
                break

        # example for simple classification
        print(f"Table: {markdown_table} classified as: {category}")

        return [{
            "model_version": GPT3,
            "result": [{
                "id": str(uuid.uuid4()),
                "from_name": "sentiment",
                "to_name": "text",
                "type": "choices",
                "value": {
                    "choices": [ category ]
                }
            }]
        }]
        
        # return ModelResponse(predictions=[])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        pass






#http://localhost:8080/api/projects/12/export?exportType=JSON&download_all_tasks=true
"""
curl -X POST "http://localhost:8080/api/predictions/" \
     -H "Authorization: Token 782ba8a7e729c9e106bf3ae69b0c217daad7103e" \
     -H "Content-Type: application/json" \
     -d '{
           "model_version": "string",
           "result": [{"from_name":"sentiment","id":"df155d8b-b239-4138-ac64-778c0ace4e7e","to_name":"text","type":"choices","value":{"choices":["unkown"]}}],
           "score": 0,
           "cluster": 0,
           "neighbors": {},
           "mislabeling": 0,
           "task": 62284,
           "project": 12
         }'
"""