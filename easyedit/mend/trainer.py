import sys
import os
sys.path.append("/data1/jutj/EasyEdit")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset

training_hparams = MENDTrainingHparams.from_hparams('./config/mend/trainer/llama.yaml')
# training_hparams = MENDTrainingHparams.from_hparams('./config/mend/trainer/qwen.yaml')
# training_hparams = MENDTrainingHparams.from_hparams('./config/mend/trainer/internlm.yaml')

train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()