from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, transformers, os


class LLMTuner(object):

    def __init__(self, settings, version=1):
        self.conf = settings

        self.model = self.load_qlora_model()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings['model_id'], token=settings['hub_id'],
            model_max_length=settings['model_max_length'],
            padding_side=settings['padding_side'],
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token


        self.version = version

    def load_qlora_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, load_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.conf['model_id'], device_map="auto",
            trust_remote_code=True, quantization_config=bnb_config,
            token=self.conf['hub_id']
        )

        lora = LoraConfig(
            r=16,  lora_alpha=64, target_modules=self.conf['target_modules'],
            lora_dropout=0.1,  bias="none", task_type="CAUSAL_LM",
        )

        return get_peft_model(prepare_model_for_kbit_training(model), lora)

    def get_traning_arguments(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        max_steps = self.conf['train'].get('max_steps', -1)
        exp_name = self.conf['tune_name']
        output_dir = f'experiments/{exp_name}_{max_steps}_{self.version}'
        

        return transformers.TrainingArguments(
            per_device_train_batch_size=self.conf['train']['batch_size'],
            per_device_eval_batch_size=self.conf['train']['batch_size'],
            num_train_epochs=self.conf['train']['epochs'],
            learning_rate=self.conf['train']['lr'], 
            max_steps=max_steps,
            output_dir=output_dir,
            lr_scheduler_type='cosine',
            eval_steps=self.conf['eval_steps'],
            save_steps=self.conf['eval_steps'], 
            warmup_steps=2,
            evaluation_strategy="steps",
            logging_strategy="steps",
            logging_steps=1,
            optim=self.conf['train']['optimizer'],
            gradient_accumulation_steps=1,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2, 
            logging_dir=output_dir
        )
    
    def train(self, train, test):
        training_args = self.get_traning_arguments()
        
        trainer = transformers.Trainer(
            model=self.model, train_dataset=train, eval_dataset=test,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        trainer.train()

        os.makedirs('trained_models', exist_ok=True)
        
        steps = self.conf['train'].get('max_steps', -1)
        name = self.conf['model_name'] +'__'+ self.conf['tune_name']

        _filename = f"trained_models/{name}-v{self.version}_{steps}"
        trainer.model.save_pretrained(_filename)