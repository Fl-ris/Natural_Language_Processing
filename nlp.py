import pickle
from collections import Counter, defaultdict
import numpy as np

from sklearn.neural_network import MLPClassifier

def file_reader(text_path):
    word_list = []
    with open(text_path) as text:
        for line in text:
            words_split = line.strip().split()
            for i in words_split:
                i = list(i.lower())
                i.append(" ") # Een spatie na elk woord, anders alle text aan elkaar vast...
                word_list.append(i)
    return word_list


def get_vocabulary(word_list):
    """
    Verkrijg alle unieke karakters in een text
    """
    vocabulary = set()
    for word in word_list:  
        for char in word:
               vocabulary.add(char)


    vocab_count = dict()
    for word in word_list:
        for char in word:
            if char in vocab_count:
                vocab_count[char] = vocab_count.get(char, 0) + 1
            else:
                vocab_count[char] = 1
    # print(vocab_count)
    return vocab_count, vocabulary

def tokenizer(word_list):
    counted_tokens = {}
    
    for word in word_list:
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            if pair in counted_tokens:
                counted_tokens[pair] += 1
            else:
                counted_tokens[pair] = 1

    #print(counted_tokens)
    return counted_tokens

def sort_and_return_token(counted_tokens, min_freq):
    """
    Neemt een dict met token counts en selecteerd te hoogste twee om in een nieuw token te veranderen.

    :param counted_tokens:
    """

    if not counted_tokens:
        return None
    
    counter_sorted = dict(reversed(list((sorted(counted_tokens.items(), key = lambda item: item[1])))))
    #print(counter_sorted)
    top = []
    for i in counter_sorted:
        top.append([i, counter_sorted[i]])

    # Neem de top twee tokens om deze te combineren:
    new_token = top.pop(0)

    # Een woord moet minimaal een n keer voorkomen voordat het een token mag worden:
    if new_token[1] < min_freq:
        return None

    return new_token[0]
            
def byte_pair_encoding(word_list, pair_to_merge):
    """Voer het byte pair encoding algorithme uit:"""
    
    new_word_list = []
    merged = pair_to_merge[0] + pair_to_merge[1]

    for word in word_list:
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word)-1 and word[i] == pair_to_merge[0] and word[i+1] == pair_to_merge[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_list.append(new_word)

    return new_word_list


            
def save_encoding(vocab, tokenized_text, enc_path="encoding.enc"):
    db = {"vocabulary": vocab, "text_tokens": tokenized_text}
    with open(enc_path, "wb") as f:
        pickle.dump(db, f)


# def save_to_file(vocabulary, text_tokens):
    
#     with open("encoding.enc", "w") as text:
#         text.write(str(vocabulary))
#     with open("story", "w") as text:
#         text.write(str(text_tokens))
    
#     db = {}
#     db['vocabulary'] = vocabulary
#     db['text_tokens'] = text_tokens
    
#     dbfile = open('Vocab_Text_Tokens', 'ab')
    
#     pickle.dump(db, dbfile)                    
#     dbfile.close()

#     pickle.dumps(vocabulary)
    
def load_encoding(enc_path):
    """Laad pickle met vocab en"""
    with open(enc_path, "rb") as f:
        return pickle.load(f)


def flatten_token_lists(token_lists):
    "Van list of lists naar list"
    return [token for word in token_lists for token in word]

def train_ngram_model(token_list, n=3):
    transitions = defaultdict(Counter)
    context = n - 1
    for i in range(len(token_list) - context):
        ctx = tuple(token_list[i : i + context])
        nxt = token_list[i + context]
        transitions[ctx][nxt] += 1

    model = {}
    for ctx, ctr in transitions.items():
        total = sum(ctr.values())
        model[ctx] = {tok: cnt / total for tok, cnt in ctr.items()}
    return model
def make_cbow_examples_ids(token_ids, window):
    """
    Bouw CBOW-training op basis van een lijst token-ID's (ints).

    input: multi-hot vector van de context (links en rechts)
    output: index van het middelste token (int)
    """
    if not token_ids:
        raise ValueError("Lege tokenlijst.")

    vocab_size = max(token_ids) + 1
    x = []
    y = []

    for i in range(window, len(token_ids) - window):
        target_id = token_ids[i]
        context_ids = token_ids[i - window:i] + token_ids[i + 1:i + 1 + window]

        x_vec = np.zeros(vocab_size, dtype=np.float32)
        for cid in context_ids:
            if 0 <= cid < vocab_size:
                x_vec[cid] = 1.0
        x.append(x_vec)
        y.append(target_id)

    x = np.vstack(x)
    y = np.array(y, dtype=np.int64)
    return x, y, vocab_size

def train_cbow_mlp(x, y, hidden_dim):
    """
    Train een simpele MLP met één verborgen laag.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation="tanh",
        solver="adam",
        max_iter=20,
        random_state=0,
    )
    clf.fit(x, y)
    return clf

def extract_embeddings_from_mlp_ids(clf):
    """
    Maak embeddings alleen voor token-ID's die het model kent als klasse.

    clf.classes_ bevat de lijst token-ID's (ints) die als y voorkwamen.
    coefs_[1] heeft shape (hidden_dim, n_classes)
    kolom j hoort bij clf.classes_[j]
    """
    w_hidden_to_out = clf.coefs_[1]
    classes = clf.classes_

    id_to_vec = {}
    for j, tid in enumerate(classes):
        vec = w_hidden_to_out[:, j]
        id_to_vec[int(tid)] = vec
    return id_to_vec


def save_embeddings_with_bpe(path, id_to_vec, bpe_tokenizer):
    """
    Schrijf embeddings weg als TSV:
    token\tval1\tval2\t...\\n

    id_to_vec: dict[int, np.ndarray]
    bpe_tokenizer: BPETokenizer met .id_to_token
    """
    with open(path, "w", encoding="utf-8") as f:
        for tid, vec in id_to_vec.items():
            if 0 <= tid < len(bpe_tokenizer.id_to_token):
                tok = bpe_tokenizer.id_to_token[tid]
            else:
                tok = f"<UNK_{tid}>"
            vals = "\t".join(str(float(x)) for x in vec)
            f.write(f"{tok}\t{vals}\n")

def save_embeddings_with_vocab(path, id_to_vec, id_to_token):
    """
    Schrijf embeddings weg als TSV:
        token<TAB>val1<TAB>...<TAB>valN
    """
    with open(path, "w", encoding="utf-8") as f:
        for tid, vec in id_to_vec.items():
            token = id_to_token[tid] if 0 <= tid < len(id_to_token) else f"<UNK_{tid}>"
            vals = "\t".join(str(float(x)) for x in vec)
            f.write(f"{token}\t{vals}\n")