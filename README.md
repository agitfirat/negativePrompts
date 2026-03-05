# NegativePrompt
Code release for [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024)

## Installation

First, clone the repo:
```sh
git clone git@https://github.com/wangxu0820/NegativePrompt
```

Then, 

```sh
cd NegativePrompt
```

To install the required packages, create the conda environment from the provided file:

```sh
conda env create -f environment.yml
conda activate chatgptTool
```

## Usage
```sh
python main.py --task task_name --model model_name --pnum negativeprompt_id --few_shot False
```


### Commandes rapides (copier/coller)
Si tu as déjà cloné le repo, place-toi dans le bon dossier (adapte le nom exact) :
```sh
cd ~/path/to/NegativePrompt-main
# ou: cd ~/path/to/NegativePrompt
pwd
```

Crée l'environnement depuis le fichier fourni par le repo :
```sh
conda env create -f environment.yml
conda activate chatgptTool
```

Tester un modèle non-GPT (exemple avec `t5`) :
```sh
python main.py --task sentiment --model t5 --pnum 0 --few_shot False
```

Lancer la même évaluation sur plusieurs modèles non-GPT :
```sh
for model in t5 vicuna llama2; do
  python main.py --task sentiment --model "$model" --pnum 0 --few_shot False
done
```

> Remarque: `vicuna` et `llama2` nécessitent des checkpoints/serveurs locaux configurés (voir `llm_response.py`).

### Open the repository in VS Code
1. Clone the project locally:
```sh
git clone https://github.com/wangxu0820/NegativePrompt.git
cd NegativePrompt
```
2. Open the folder in VS Code:
```sh
code .
```
3. If `code` is not available in your shell, open VS Code and use **File > Open Folder...** then select `NegativePrompt`.

### Run evaluations with non-GPT models
Supported non-GPT options in this repository are currently `t5`, `vicuna`, and `llama2`.

Example command:
```sh
python main.py --task sentiment --model t5 --pnum 0 --few_shot False
```

Run the same task across all non-GPT models:
```sh
for model in t5 vicuna llama2; do
  python main.py --task sentiment --model "$model" --pnum 0 --few_shot False
done
```

You can replace `sentiment` by any available task (for example: `sum`, `word_in_context`, `translation_en-fr`).

## Citation
Please cite us if you find this project helpful for your research:
```
@misc{wang2024negativeprompt,
      title={NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli}, 
      author={Xu Wang and Cheng Li and Yi Chang and Jindong Wang and Yuan Wu},
      year={2024},
      eprint={2405.02814},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
