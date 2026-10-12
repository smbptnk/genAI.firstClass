import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
except Exception:  # pragma: no cover - optional runtime dependency
    Credentials = None
    ModelInference = None

FILEPATH = os.path.join(os.getcwd(), 'structured_restaurant_data.json')
BACKUP_PATH = os.path.join(os.getcwd(), 'structured_restaurant_data.json.bak')
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $'


def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """Build the system/user prompt used to extract a structured restaurant record."""
    example_output = """
    {
      "name": "Mar de Cortez",
      "location": "Santa Monica",
      "type": "casual taqueria",
      "food_style": "Baja-style seafood",
      "rating": 4.2,
      "price_range": 1,
      "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
      ],
      "vibe": "salt-air energy",
      "environment": "a premier sun-drenched spot for open-air dining near the pier.",
      "shortcomings": []
    }
    """

    base_system_msg = """
        You are a data extraction assistant that converts restaurant descriptions into structured JSON format.
        Your task is to extract specific information from a restaurant paragraph and format it consistently.
    """

    base_user_prompt = f"""
    Task:
    Extract the following information from the restaurant description and format it as a JSON object:
    - name: The restaurant name
    - location: The city or neighborhood where the restaurant is located
    - type: The type of restaurant (e.g., casual taqueria, fine dining, food truck)
    - food_style: The cuisine style or type of food served
    - rating: The numerical rating (if mentioned, otherwise use 0.0)
    - price_range: Convert dollar signs ($, $$, $$$) to an integer representing the number of dollar symbols (e.g., $ -> 1, $$ -> 2, $$$ -> 3)
    - signatures: A list of signature dishes or popular menu items (extract 2-3 items)
    - vibe: The atmosphere or feeling of the restaurant
    - environment: A complete sentence describing the setting and ambiance
    - shortcomings: A list of any negative aspects mentioned (if none, return an empty list)

    Important rules:
    1. Output must be valid JSON format
    2. If a field is not explicitly mentioned, use reasonable inference from context
    3. For price_range, count the number of dollar signs ($) in the description and convert to integer
    4. For signatures, extract specific dish names mentioned in the text
    5. For shortcomings, only include explicitly mentioned negatives

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {example_output}

    Now extract the information from the provided restaurant description following the same format.
    """
    return base_system_msg, base_user_prompt


def llm_model(system_msg, prompt_txt, params=None):
    """Call the IBM Granite model with a system+user prompt and return the raw reply."""
    if Credentials is None or ModelInference is None:
        raise RuntimeError(
            'IBM watsonx dependencies are not available in this environment. '
            'Install the package and configure credentials before calling llm_model().'
        )
    
    #YOUR CODE HERE
    """
    Call the Granite LLM with a system + user message and return the
    raw response text.

    Granite is it is a small, cost-efficient model
    that is sufficient for structured extraction tasks.
    """
    model_id = 'ibm/granite-4-h-small'
    project_id = 'skills-network'
    credentials = Credentials(url='https://us-south.ml.cloud.ibm.com')

    if params is None:
        params = {'max_tokens': 500, 'temperature': 0.0}

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params,
    )

    messages = [
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': prompt_txt},
    ]

    response = model.chat(messages=messages)
    if isinstance(response, dict):
        return response['choices'][0]['message']['content']
    return response


def JSON_auto_repair_prompts(candidate_json_output, error_message):
    """Build a repair prompt for malformed JSON output."""
    missing_inputs = ''
    auto_repair_system_msg = f"""
    You are a JSON repair specialist. Your sole responsibility is to fix malformed JSON outputs
    from restaurant data extraction. You receive candidate JSON outputs that failed validation
    along with specific error messages, and you must correct them to produce valid JSON that
    conforms to the required schema.

    Your expertise includes:
    - Fixing JSON syntax errors (missing commas, brackets, quotes)
    - Correcting data type issues (strings where numbers should be, etc.)
    - Ensuring all required fields are present
    - Maintaining the semantic meaning of the original data
    - Outputting ONLY valid JSON without any additional text or explanation
    Missing information: {missing_inputs}
    """

    auto_repair_prompt = f"""
    TASK: Repair the following JSON output so it validates correctly against our schema.

    ORIGINAL (INVALID) OUTPUT:
    {candidate_json_output}

    VALIDATION ERROR MESSAGE:
    {error_message}

    REQUIRED JSON SCHEMA:
    {{
        "name": string,
        "location": string,
        "type": string,
        "food_style": string,
        "rating": float,
        "price_range": integer (1-3),
        "signatures": list of strings (2-3 items),
        "vibe": string,
        "environment": string (complete sentence),
        "shortcomings": list of strings (empty list if none)
    }}

    REPAIR GUIDELINES:
    1. Fix any JSON syntax errors indicated in the error message
    2. Ensure all fields from the schema are present
    3. Correct data types to match the schema (e.g., strings in quotes, numbers without quotes)
    4. For price_range, ensure it's an integer (1, 2, or 3)
    5. For rating, ensure it's a float (e.g., 4.2, not "4.2" or 4)
    6. For signatures and shortcomings, ensure they are arrays (lists)
    7. Preserve the original content and meaning as much as possible
    8. If a field is missing entirely, infer it from context or use reasonable defaults (name: "Unknown", location: "Unknown", rating: 0.0)
    9. Output ONLY the repaired JSON - no explanations, no markdown formatting, no additional text
    10. The repaired JSON must be valid and parseable

    REPAIRED JSON OUTPUT:
    """

    return auto_repair_system_msg, auto_repair_prompt


def load_data(file_path):
    """Return the list of restaurant records from disk, or [] if the file is missing."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_data(file_path, data=None):
    """Save a list of restaurant records to disk, accepting either argument order."""
    if data is None and isinstance(file_path, list):
        data = file_path
        file_path = FILEPATH
    elif data is not None and not isinstance(file_path, str):
        file_path, data = data, file_path

    if data is None:
        data = []

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def show_restaurant_card(record, index):
    """Pretty-print one restaurant record."""
    if not isinstance(record, dict):
        print(f"Record {index} is not valid.")
        return

    print(f"\n--- Restaurant #{index} ---")
    for key, value in record.items():
        print(f"{key}: {value}")


def new_data_entry_process(paragraph, itemId):
    """Convert a restaurant paragraph into a structured JSON format and append it to the data file."""
    if not isinstance(paragraph, str) or not paragraph.strip():
        return None

    system_msg, prompt_txt = restaurant_data_structure_prompt_generation(paragraph)
    raw_response = llm_model(system_msg, prompt_txt)
    structured = None
    max_repair_attempts = 3

    for attempt in range(max_repair_attempts + 1):
        try:
            structured = json.loads(raw_response)
            break
        except json.JSONDecodeError as e:
            if attempt == max_repair_attempts:
                print(f'Failed to repair JSON after {max_repair_attempts} attempts: {e}')
                return None
            repair_sys, repair_txt = JSON_auto_repair_prompts(raw_response, str(e))
            raw_response = llm_model(repair_sys, repair_txt)

    if not isinstance(structured, dict):
        print('LLM did not return a JSON object.')
        return None

    structured['itemId'] = itemId

    if os.path.exists(FILEPATH):
        shutil.copy2(FILEPATH, BACKUP_PATH)
        data = load_data(FILEPATH)
    else:
        data = []

    data.append(structured)
    save_data(FILEPATH, data)
    return structured


def manage_restaurants(file_path=FILEPATH, backup_path=BACKUP_PATH):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print('1. Browse All (Names)')
        print('2. View Detailed Record')
        print('3. Add New Restaurant')
        print('4. Edit Restaurant Info')
        print('5. Delete Restaurant')
        print('6. Exit')

        choice = input('\nAction: ')

        if choice == '1':
            print('\n--- Current Listings ---')
            for idx, restaurant in enumerate(data):
                print(f'{idx}: {restaurant.get("name", "N/A")}')

        elif choice == '2':
            idx = _valid_index(input('Enter record index: '), data)
            if idx is not None:
                show_restaurant_card(data[idx], idx)
            else:
                print('Invalid index.')

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print('\n❗ SECURITY WARNING: You are entering write-mode.')
            print('Changes will be saved to the database immediately.')
            confirm = input('Are you sure? (type \'yes\' to proceed): ').lower()
            if confirm != 'yes':
                print('Operation cancelled.')
                continue

            if choice == '3': # ADD NEW DATA
                item_id = 1000000 + len(data) + 1 #the item id for the new data

                # YOUR CODE HERE
				# Instruction:
				# First: ask the user to input a new restaurant description.
				# Second: use new_data_entry_process() to process the new paragraph.
				# Third: append the new restaurant data to the original data list.
				# Finally: save using save_data().
                paragraph = input('Enter new restaurant description:\n').strip()
                if not paragraph:
                    print('Empty input — cancelled.')
                    continue

                new_restaurant = new_data_entry_process(paragraph, item_id)
                if new_restaurant is None:
                    print('Restaurant could not be added.')
                    continue

                data = load_data(file_path)
                print('✅ Restaurant added.')

            elif choice == '4': # EDIT DATA
                # YOUR CODE HERE
				# Instruction:
				# First: ask for the input record index.
				# Second: iterate over the keys of the current record, and ask for
				#         new values. If the user doesn't want to update, a simple 
				#         Enter can skip. Update it only when the input index is 
				#         valid.
				# Third: save with save_data() and notify ("✅ Record updated.")
                idx = _valid_index(input('Enter record index to edit: '), data)
                if idx is None:
                    print('invalid index.')

                record = data[idx]
                print('Press Enter to keep current value.')
                for key in list(record.keys()):
                    new_val = input(f'  {key} [{record[key]}]: ').strip()
                    if new_val:
                        record[key] = _coerce(new_val, record[key])
                data[idx] = record
                save_data(file_path, data)
                print('✅ Record updated.')

            elif choice == '5':
                idx = _valid_index(input('Enter record index to delete: '), data)
                if idx is None:
                    print('invalid index.')
                    continue
                deleted = data.pop(idx)
                save_data(file_path, data)
                print(f"✅ Deleted: {deleted.get('name', 'N/A')}")

        elif choice == '6':
            break
        else:
            print('Invalid input.')


# Helper functions

def _valid_index(user_input, data):
    """Return a valid zero-based index, else None."""
    try:
        ids = list(map(lambda record: record["id"], data))
        if user_input in ids:
            return int(user_input)
    except (ValueError, TypeError):
        return None


def _coerce(new_value, current_value):
    if isinstance(current_value, bool):
        return new_value.lower() in ('true', '1', 'yes')
    for typ in (int, float):
        if isinstance(current_value, typ):
            try:
                return typ(new_value)
            except ValueError:
                return current_value
    if isinstance(current_value, list):
        return [v.strip() for v in new_value.split(',') if v.strip()]
    return new_value

class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn('✅ Restaurant added.', mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', '1', '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn('Operation cancelled.', mock_stdout.getvalue())


if __name__ == '__main__':
    # unittest.main()
    # manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
    manage_restaurants()