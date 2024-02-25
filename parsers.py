import json, re


_parser = r = re.compile(r"[^a-zA-Z0-9:]")

def cleaner_llama_json(text):
    text = text.split('{')[1]
    return '{' + text.split('}')[0] + '}'


def cleaner_mistral_json(text):
    text = text.replace('json output:', '')
    text = text.replace('json:', '')
    return text


cleaner_functor = {
    'mistral': cleaner_mistral_json, 
    'llamav2': cleaner_llama_json
}

def plain_two_level_text_to_json(text):
    
    output = {}
    
    for x in text.split('\n'):
        if len(x) > 5:
            parts = x.split(':')
            if len(parts) != 2:
                break
            key = parts[0].strip().replace('"', '')
            pred = parts[1].strip().replace(
                '"', ''
            ).replace(',', '').replace('.', '')
            output[key] = pred
    
    return output


def force_json(text):
    try:
        if '{' in text or '}' in text:
            text = _parser.sub(' ', text).strip().replace('   ', '\n')
            return plain_two_level_text_to_json(text)
        else:
            return plain_two_level_text_to_json(text)
    except: pass

    return {}


def cast_str_to_json(text, model):
    """_summary_

    Args:
        text (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return json.loads(cleaner_functor[model](text))
    except: pass
    
    return force_json(text)


def get_prediction(_json_data):
    try:
        if _json_data.get('prediction'):
            return _json_data['prediction']
        return _json_data['explanation']
    except: pass
