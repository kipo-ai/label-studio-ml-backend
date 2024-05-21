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

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")


    def create_classification_prompt(self, markdown_table: str) -> str:
        """
        Generates a prompt for classifying the table from technical datasheets.
        """
        intro = """
        Analyze the following markdown table extracted from a technical datasheet of electronic components:\n\n
        {markdown_table}\n\n
        Classify the table into one of the following categories based on its content and structure:\n\n
        """

        # Section 2: Common Table Types
        common_tables = """
        - **table_of_contents:** Lists sections and their page numbers within the datasheet (non-technical).\n
        * Look for headings, subheadings, numbers, and page references.\n\n
        
        - **pin_table:**  Details pin numbers, functions, and descriptions (crucial for hardware design).\n
        * May contain terms like 'Pin,' 'Terminal,' 'Signal,' or 'Ball.'\n
        * Pin numbers often follow this pattern: '^[a-zA-Z]{0,2}[0-9]{1,2},?$'\n\n
        * *Example of pin tables:*
        
        | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
        |:-----------|:----------------|:----------------|:----------------|:---------------------|:--------------|:---------------------------|
        | PIN NUMBER | SIGNAL NAME (1) | SIGNAL TYPE (2) | BUFFER TYPE (3) | PIN MUX ENCODING (4) | POWER SOURCE5 | STATE AFTER RESET RELEASE6 |
        | | PG5 | I/O | LVCMOS | - | | OFF |
        | | EN0TXD1 | O | LVCMOS | PG5 (14) | | N/A |
        | K15 | 12C3SDA | I/O | LVCMOS | PG5 (2) | | N/A |

        | 0 | 1 | 2 | 3 |
        |:------|:----|:----|:--------------------------------|
        | PIN | | I/O | DESCRIPTION |
        | NAME | NO. | | |
        | +IN A | 3 | I | Noninverting input, channel A |
        
        - **pin_map:**  Visual representation of pin layout (often a diagram).\n
        * Primarily contains pin names and numbers as headers.\n
        * Letters and numbers in headers may be orthogonal (independent).\n\n
        * *Example of pin maps:*
        | 1  | 2        | 3         | 4         | 5   | 6       | 7   |
        |----|----------|-----------|-----------|-----|---------|-----|
        |DVDD| DVDD_SD  | LDOCAP_DGP| VODA_DSPFL| CVDD| VSS     |CVDD |
        | VSS| CVDD_DSP | LDOCAP_JC | LDOCAP_VSS|     | CVDD_HM |     |
        |    | DVDD     | VSS       | CVDD_DSP  |     | CVDD_HM |     |
        | VSS| CVDD_DSP | CVDD_DSP  | CVDD_DSP  |     |         |     |
        |    | DVDD     |           |           |     |         |     |
        | ...| ...      | ...       | ...       | ... | ...     | ... |

        
        - **electrical_specifications_table:** Key technical parameters (voltage, current, etc.).\n
        * Focus on electrical characteristics and operating conditions.\n
        * Typically includes parameters, test conditions, min/typ/max values, and units.
        * *Example of electrical specifications tables:*

        | 0 | 1 | 2 | 3 | 4 |
        |:-----------|:----------|:--------------------|:-----------------|:--------|
        | | PARAMETER | TEST CONDITIONS | MIN TYP MAX | UNIT |
        | OFFSET | VOLTAGE | | | |
        | Vos | Input offset voltage | Vs=5 V | +0.33 +1.6 | mV |
        | | | Vs = 5 V, TA = -40°C to +125°C | +2 | |
        | dVos/dT | Drift | Vs = 5 V, TA = -40°C to +125°C | +0.5 | uV/°C |
        | PSRR | Power-supply rejection ratio | Vs = 1.8 V - 5.5 VCM = (V-) | +13 +80 | uV/V |
        | ... | ... | ... | ... | ... |

        - **package_information_table:** Physical packaging details (size, pin count, form factor).\n\n
        * Contains dimensions, materials, and other physical characteristics.\n
        * *Example of package information table:*

        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        |:----------------|:-------------|:----------------|:-----|:-----|:------------|:-----------|:------------|
        | Device | Package Type | Package Drawing | Pins | SPQ | Length (mm) | Width (mm) | Height (mm) |
        | LM124DR | SOIC | D | 14 | 2500 | 350.0 | 350.0 | 43.0 |
        | ... | ... | ... | ... | ... | ... | ... | ... |

        - **dimensional_table:** Measurements, dimensions, and tolerances (for mechanical design).\n\n
        """

        # Section 3: Other (non-standard) Tables
        other_tables = """
        - **other:**  Tables not fitting the common categories.  This can include:
        * Register tables listing addresses, names, values
        * Bit address mapping tables
        * Address lookup tables
        * Interrupt tables
        * Timing diagrams
        * Control registers
        * Status registers
        * Revision tables
        * Any table with unclear content or purpose\n\n
        
        * *Examples of 'other' tables:**

        | 0 | 1 | 2 |
        |:----------------|:------------|:-------------------|
        | REGISTER NUMBER | RESET VALUE | REGISTER NAME |
        | 19 | 11101111 | 8 LSBs of d1 coefficient for DRC first-order high-pass filter |
        | 20 | 0000 0000 | 8 MSBs of n0 coefficient for DRC first-order low-pass filter |
        | 21 | 0001 0001 | 8 LSBs of n0 coefficient for DRC first-order low-pass filter |
        | ... | ... | ... |


        | 0 | 1 | 2 |
        |:-------------|:---------|:-----------------|
        | HEX ADDRESS | ACRONYM | REGISTER NAME |
        | 0x4808 0000 | ELM_REVISION | Revision |
        | 0x4808 0010 | ELM_SYSCONFIG | Configuration |
        | 0x4808 0014 | ELM_SYSSTATUS | Status |
        | ... | ... | ... |


        | 0 | 1 | 2 | 3 |
        |:------|:----------|:---------|:----------|
        | BIT | READ/ WRITE | RESET VALUE | DESCRIPTION |
        | D7 | R/W | 1 | 0: ADC channel not muted 1: ADC channel muted |
        | ... | ... | ... | ... |

        """

        not_a_table = """
        - **not_a_valid_table:**  The content is not a table or does not fit any of the above categories.\n
        * This can include images, diagrams, text, or any other non-tabular content.\n\n
        * **Example:**

        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
        |:----|:---------|:----|:----|:----|:----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|
        | | X + + A | | | | | | | | | | | | |
        | | X+ B | | | | | | | | | | | | |
        | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

        | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
        |:-----|:-----|:-----|:----|:-------|:-------|:-----|
        | 1,1 | 2, 1 | 3, 1 | | P-2, 1 | P-1, 1 | P, 1 |
        | 1,2 | 2, 2 | | | | P-1, 2 | P, 2 |
        """

        # Section 4: Response Instructions
        response_format = """

        Respond with the most appropriate category in JSON format:\n\n
        {\n
        "category": "your_category_here"\n  
        }\n\n
        Ensure the 'category' value matches one of the listed options.
        """

        return intro + common_tables + other_tables + not_a_table + response_format.format(markdown_table=markdown_table) 

    
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


class TableModel(NewModel):

    def __init__(self):
        pass