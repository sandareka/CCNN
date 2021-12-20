import argparse
import numpy as np
import cub_params as params

#######
# Preparing data files (indicator vector, class indicator vector and concept word phrase vectors)
#######

def create_concept_list(image_word_presence, classes, class_list):
    class_word_phrase_occurrences = {}
    for key, part_desc in image_word_presence.items():

        class_name = "_".join(key.lower().split("_")[:-2])
        if class_name not in class_word_phrase_occurrences.keys():
            class_word_phrase_occurrences[class_name] = {}

        for noun, adjectives in part_desc.items():

            if noun == 'body':
                continue

            for adjective in list(adjectives):
                if adjective + " " + noun not in class_word_phrase_occurrences[class_name].keys():
                    class_word_phrase_occurrences[class_name][adjective + " " + noun] = 1
                else:
                    class_word_phrase_occurrences[class_name][adjective + " " + noun] = class_word_phrase_occurrences[class_name][adjective + " " + noun] + 1

    high_frequent_concepts = set()
    new_class_features = {}
    for key, value in class_word_phrase_occurrences.items():
        new_class_features[key] = {}
        for sub_key, sub_value in value.items():
            if sub_value < 3:
                continue
            high_frequent_concepts.add(sub_key)

    concepts = np.asarray(list(high_frequent_concepts))
    num_concepts = len(high_frequent_concepts)
    class_concept_count = np.zeros(shape=[params.NO_CLASSES, len(high_frequent_concepts)])

    for key, value in class_word_phrase_occurrences.items():
        class_id = classes[key]
        for sub_key, sub_value in value.items():
            if sub_key not in list(high_frequent_concepts):
                continue
            if sub_value < 3:
                continue
            index = np.where(concepts == sub_key)[0][0]
            class_concept_count[class_id, index] = sub_value

    class_concept_tf = np.divide(class_concept_count, np.tile(np.sum(class_concept_count, axis=1, keepdims=True), [1, num_concepts]))
    class_concept_idf = np.log(np.divide(np.tile(np.array(params.NO_CLASSES), [num_concepts]), np.count_nonzero(class_concept_count > 0, axis=0)))
    class_tf_idf = np.multiply(np.tile(np.expand_dims(class_concept_idf, axis=0), [params.NO_CLASSES, 1]), class_concept_tf)

    class_discriminative_concepts = {}
    selected_concepts = set()
    for k in range(params.NO_CLASSES):
        tf_idf_scores = class_tf_idf[k, :]
        tf_idf_scores = np.sort(tf_idf_scores)[::-1]
        tf_idf_scores.sort()
        tf_idf_scores = tf_idf_scores[::-1]
        index_sm = np.where(tf_idf_scores == 0)[0][0]

        if index_sm > params.TOP_R:
            index_sm = params.TOP_R

        class_discriminative_concept_set = concepts[class_tf_idf[k, :].argsort()[::-1][:index_sm]]
        class_discriminative_concepts[class_list[k]] = set()
        for r in range(len(class_discriminative_concept_set)):
            if 'and' in class_discriminative_concept_set[r]:
                reversed_phrase = class_discriminative_concept_set[r].split(' ')[2] + ' and ' + class_discriminative_concept_set[r].split(' ')[0] + ' ' + \
                                  class_discriminative_concept_set[r].split(' ')[3]
                if reversed_phrase in selected_concepts:
                    class_discriminative_concepts[class_list[k]].add(reversed_phrase)
                    continue

            class_discriminative_concepts[class_list[k]].add(class_discriminative_concept_set[r])
            selected_concepts.add(class_discriminative_concept_set[r])

    selected_concepts_list = list(selected_concepts)
    selected_concepts_list.sort()

    return selected_concepts_list, class_discriminative_concepts


def create_class_and_image_indicator_vectors(concepts, class_discriminative_concepts, classes):
    image_indicator_vector = {}
    class_indicator_vector = np.zeros(shape=[len(concepts), params.NO_CLASSES])
    image_word_presence = np.load(params.NOUNS_ADJECTIVES_FILE, allow_pickle=True).item()

    for key, part_desc in image_word_presence.items():

        word_phrases = set()
        image_indicator_vector[key] = np.zeros(shape=len(concepts))
        class_name = "_".join(key.lower().split("_")[:-2])
        class_id = classes[class_name]

        for noun, adjectives in part_desc.items():

            if noun == 'body':
                continue

            for adjective in list(adjectives):
                word_phrases.add(adjective + " " + noun)

        class_related_features = class_discriminative_concepts[class_name]
        word_phrases = list(word_phrases)
        for i in range(len(word_phrases)):
            if word_phrases[i] in class_related_features:
                concept_index = concepts.index(word_phrases[i])
                image_indicator_vector[key][concept_index] = 1
                class_indicator_vector[concept_index][class_id] = 1

    np.save(params.INDICATOR_VECTORS, image_indicator_vector)
    np.save(params.CLASS_INDICATOR_VECTORS, class_indicator_vector)


def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        words = line.split()
        word = words[0]
        if word == 'eye-ring':
            word = 'eyering'
        embedding = np.array([float(val) for val in words[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def create_word_phrase_vectors(concepts):
    model = load_glove_model(params.GLOVE_FILE_NAME)

    phrase_vectors = np.zeros(shape=(len(concepts), params.TEXT_SPACE_DIM))
    for i in range(len(concepts)):
        words = concepts[i].split(' ')
        vector_value = 0
        for word in words:
            if word == 'and':
                continue
            if 'wingbar' == word:
                vector_value = vector_value + model['wing'] + model['edge']
                continue
            if 'cheek-patch' == word:
                vector_value = vector_value + model['cheek'] + model['patch']
                continue
            vector_value = vector_value + model[word]
        phrase_vectors[i, :] = vector_value

    np.save(params.CONCEPT_WORD_PHRASE_VECTORS, phrase_vectors)


def prepare():
    classes = {}
    class_no = 0
    class_list = []
    with open(params.CLASS_NAME_FILE, 'r') as f:
        for line in f.readlines():
            class_name = line.rstrip("\n\r").split(".")[1].lower()
            classes[class_name] = class_no
            class_list.append(class_name)
            class_no += 1

    # nouns and their associated adjectives for each image in the training dataset. This is created by going through
    # the text description of images and extracting nouns and associated nouns using an algorithm based on NLTK library.

    image_word_presence = np.load(params.NOUNS_ADJECTIVES_FILE, allow_pickle=True).item()

    # Create the set of concept CCNN is supposed to learn in concept-layer
    concepts, class_discriminative_concepts = create_concept_list(image_word_presence, classes, class_list)

    create_class_and_image_indicator_vectors(concepts, class_discriminative_concepts, classes)

    create_word_phrase_vectors(concepts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="prepare")
    args = parser.parse_args()
    if args.method == 'prepare':
        prepare()

