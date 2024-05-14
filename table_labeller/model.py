from enum import Enum
import json
from math import exp
import os
from typing import Any, List, Dict, Optional, Tuple, Union
import uuid

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from constants.LLMConstants import GPT3
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
    
    def __init__(self, **kwargs):
        """Initialize the model
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
            "Table of contents usually have section, subsections with corresponding numbers and page numbers.\n\n"
            "- pin_table: Contains details such as pin numbers, functions, and descriptions; crucial for hardware design.\n"
            "Pin tables usually contains information about pins. With headers like Pin number, name, description etc.\n"
            "Pin is also represented as Terminal, Signal or Ball. Pin Number follows this regex: '^[a-zA-Z]{0,2}[0-9]{1,2},?$'\n\n"
            "- pin_map: Provides a graphical or schematic representation of pin layouts; usually accompanied by diagrams.\n"
            "Pin maps usually contains pin names only with Pin Number as headers.\n"
            "Pin number 'alphabet (^[a-zA-Z])' and 'digits( {0,2}[0-9]{1,2},?$)' are orthogonal headers. Sometimes they are not present in the table.\n\n"
            "- specs_table: Outlines technical specifications like voltage, current, dimensions, and operating parameters.\n\n"
            "- package_information_table: Includes details about the physical packaging of the component like size, pin count, and form factor.\n\n"
            "- not_a_valid_table: The content does not represent a structured table; may include random text or images.\n\n"
            "- dimensional_table: Contains measurements, dimensions, and tolerances; essential for mechanical design.\n\n"
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
