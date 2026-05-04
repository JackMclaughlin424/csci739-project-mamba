from dotenv import load_dotenv
from experiments.icl_task_vectors.core.experiments_config import MODELS_TO_EVALUATE

load_dotenv(".env")

from experiments.icl_task_vectors.core.models.llm_loading import load_model_and_tokenizer


def main():
    for model_params in MODELS_TO_EVALUATE:
        print("Downloading", model_params)
        load_model_and_tokenizer(*model_params, load_to_cpu=True)


if __name__ == "__main__":
    main()
