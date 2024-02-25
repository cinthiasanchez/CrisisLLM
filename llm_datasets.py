from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk
import pandas as pd, pathlib, os, json
from transformers import AutoTokenizer
import constants


class LLMDatabaseBuilder(object):
    
    def __init__(self, settings, version=1):
        
        self.load_databases(settings)
        self.model_format = getattr(constants, settings['model_name'])
        self.conf, self.version = settings, version
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings['model_id'], token=settings['hub_id'],
            model_max_length=settings['model_max_length'],
            padding_side=settings['padding_side'],
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.source_data = settings['tune_name']
        self.promt_base, self.promt_args = self.load_promt()
        self.set_endpoint()

    def load_databases(self, settings):
        """_summary_

        Args:
            settings (_type_): _description_
        """

        if isinstance(settings['database'], dict):
            dataframes = []
            for task, filename in settings['database'].items():
                subset = pd.read_csv(filename)
                subset['task'] = task
                dataframes.append(subset)
            self.dataframe = pd.concat(dataframes, ignore_index=True)
        else:   
            self.dataframe = pd.read_csv(settings['database'])
            self.dataframe['task'] = settings['tune_name']
        
        self.dataframe.fillna('', inplace=True)
        self.dataframe = self.dataframe.sample(
            n=len(self.dataframe), random_state=7
        )

    def set_endpoint(self):
        self.endpoint = pathlib.Path().home() / self.conf['datasets']
        if self.endpoint.exists():
            os.makedirs(self.endpoint, exist_ok=True)

    def load_promt(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        args = {}

        if isinstance(self.conf["k-shot"], dict):
            for task, _args_task in self.conf['k-shot'].items():
                args[task] = self.load_args_promt(_args_task)
        else:
            args[self.conf['tune_name']] = self.load_args_promt(self.conf["k-shot"])

        return getattr(constants, self.conf['promt_template']), args

    def load_args_promt(self, promt_name):
        with open(f'{self.conf["promt_source"]}/{promt_name}') as buffer:
            args = json.load(buffer)[self.conf['k-selected']]
        
        return args

    def tokenize(self, data_point):
        full_prompt = f"""{data_point["text"]}""".strip()
        return self.tokenizer(full_prompt, padding=True, truncation=True)

    def join_dataframe_and_prompt(self, data_point):
        system_promt = dict(model=self.promt_args[data_point['task']]['model'])

        system_promt.__setitem__('task', self.promt_args[data_point['task']]['task'])
        system_promt.__setitem__(
            'categories_description', 
            self.promt_args[data_point['task']]['categories_description']
        )
        system_promt_txt = self.promt_base.format(**system_promt)

        user_promt = dict(
            category_list=self.promt_args[data_point['task']]['category_list']
        )
        
        user_promt.__setitem__('tweet', data_point['tweet'])
        user_promt_txt = constants.QUESTION.format(**user_promt)

        model_answer = ""
        
        if self.conf['add_answer'] and bool(self.conf['add_explanation']):
            model_answer = json.dumps({
                "prediction": data_point['label'], 
                "explanation":  data_point['explanation']
            })
        elif self.conf['add_answer'] and not bool(self.conf['add_explanation']):
            user_promt_txt = constants.QUESTION.format(**user_promt)
            model_answer = json.dumps({"prediction": data_point['label']})

        args = dict(
            system_promt=system_promt_txt, user_promt=user_promt_txt,
            model_answer=model_answer
        )

        return self.model_format['input'].format(**args)

    
    def generate_db(self, dataset, source='train'):
        dataset_tokens = dataset.map(self.tokenize)
        dataset_tokens.save_to_disk(
            self.endpoint / f'{source}_v{self.version}_{self.source_data}'
        )
        return dataset_tokens

    def build_dataset(self):
        """_summary_
        """
        prompt = self.dataframe.apply(
            getattr(self, self.conf['base_promt']), axis=1
        )
        
        train, test =  train_test_split(
            prompt, test_size=self.conf['test_size'], 
            stratify=self.dataframe['label'],
            random_state=self.conf['random_state_partition']
        )

        train_db = Dataset.from_dict({'text': train.tolist()})
        test_db = Dataset.from_dict({'text': test.tolist()})

        return self.generate_db(train_db), self.generate_db(test_db, source='test')

    def get_datasets(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if (self.endpoint / f'train_v{self.version}_{self.source_data}').exists():
            train =  load_from_disk(
                self.endpoint / f'train_v{self.version}_{self.source_data}'
            )
            test =  load_from_disk(
                self.endpoint / f'test_v{self.version}_{self.source_data}'
            )
            return train, test
        return self.build_dataset()
