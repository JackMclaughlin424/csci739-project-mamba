from icl_task_vectors.core.data.preparation.knowledge_data import prepare_knowledge_data
from icl_task_vectors.core.data.preparation.linguistic_data import prepare_linguistic_data
from icl_task_vectors.core.data.preparation.translation_data import prepare_translation_data


def prepare_data():
    print("Preparing knowledge data...")
    prepare_knowledge_data()
    print("Preparing linguistic data...")
    prepare_linguistic_data()
    print("Preparing translation data...")
    prepare_translation_data()
    print("Done.")


if __name__ == "__main__":
    prepare_data()
