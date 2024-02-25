llamav2 = {
    'input': "<s>[INST] <<SYS>> {system_promt} <</SYS>> {user_promt} [/INST] {model_answer} </s>"
}


PROMT_TEMPLATE = """You are a {model}. {task}: {categories_description} The tweet text is delimited with triple backticks."""


QUESTION = """
Generate a JSON with the following keys: 'prediction' containing a single predicted category, either {category_list}, and 'explanation' containing the reason (limited to 100 characters) for the categorization decision.
Tweet text: '''{tweet}'''
"""
