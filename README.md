# NegativePrompt
Code release for [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024)

## Modèles utilisés

| Modèle | Identifiant HuggingFace | GPU requis |
|--------|------------------------|------------|
| T5 | `google/flan-t5-large` | ~3 GB VRAM |
| Vicuna | `lmsys/vicuna-13b-v1.5` | ~28 GB VRAM (fp16) |
| Llama2 | `meta-llama/Llama-2-13b-chat-hf` | ~28 GB VRAM |

> **Note Vicuna** : Le papier original utilise `vicuna-13b-v1.1` via un serveur FastChat local. Cette version utilise `lmsys/vicuna-13b-v1.5` directement via HuggingFace transformers, ce qui donne des résultats équivalents.

> **Note Llama2** : L'accès au modèle `meta-llama/Llama-2-13b-chat-hf` nécessite d'accepter la licence sur HuggingFace puis de se connecter (voir ci-dessous).

## Installation

```sh
git clone https://github.com/ac2408/negativePrompts
cd negativePrompts
```

Créer l'environnement conda :

```sh
conda env create -f environment.yml
conda activate chatgptTool
```

Installer les dépendances supplémentaires :

```sh
pip install -r requirements.txt
```

## Lancer sur Kaggle Notebook

```python
import subprocess, os

# Cloner le repo (à lancer une seule fois)
if not os.path.exists("/kaggle/working/negativePrompts"):
    subprocess.run(["git", "clone", "-b", "branche_chen",
                    "https://github.com/ac2408/negativePrompts",
                    "/kaggle/working/negativePrompts"], check=True)

os.chdir("/kaggle/working/negativePrompts")  # chemin absolu, stable entre cellules

# Installer les dépendances
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=True)

# Lancer une évaluation T5 (pas besoin de token HuggingFace)
result = subprocess.run(
    ["python", "main.py", "--task", "sentiment", "--model", "t5", "--pnum", "0", "--few_shot", "False"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
```

Pour Llama-2, ajouter avant le run :
```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

## Prérequis pour Llama-2

Accepter la licence sur https://huggingface.co/meta-llama/Llama-2-13b-chat-hf puis s'authentifier :

```sh
pip install huggingface_hub
huggingface-cli login
```

## Usage

```sh
python main.py --task sentiment --model t5 --pnum 0 --few_shot False
```

**Paramètres :**
- `--task` : nom de la tâche (ex: `sentiment`, `larger_animal`, `antonyms`, ...)
- `--model` : `t5`, `vicuna`, `llama2`, `chatgpt`, `gpt4`
- `--pnum` : `0` = prompt original, `1` à `10` = NP01 à NP10
- `--few_shot` : `True` ou `False`

**Exemple pour tester tous les prompts négatifs (NP01–NP10) sur sentiment avec T5 :**

```sh
for i in $(seq 0 10); do
    python main.py --task sentiment --model t5 --pnum $i --few_shot False
done
```

## Tâches disponibles

```
active_to_passive, antonyms, cause_and_effect, common_concept, diff,
first_word_letter, informal_to_formal, larger_animal, letters_list,
negation, num_to_verbal, orthography_starts_with, rhymes,
second_word_letter, sentence_similarity, sentiment, singular_to_plural,
sum, synonyms, taxonomy_animal, translation_en-de, translation_en-es,
translation_en-fr, word_in_context
```

## Citation

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
