from helper.functions import *
from helper.libraries import *


# Distractors from Wordnet
def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        # print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [(gloss, 1 if i in record.targets else 0) for i, gloss in enumerate(record.glosses)]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label)
            )

        features.append(pairs)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_sense(sent):
    re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")

    ambiguous_word = re_result.group(1).strip()

    results = dict()

    wn_pos = wn.NOUN
    for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
        results[synset] = synset.definition()

    if len(results) == 0:
        return (None, None, ambiguous_word)

    # print (results)
    sense_keys = []
    definitions = []
    for sense_key, definition in results.items():
        sense_keys.append(sense_key)
        definitions.append(definition)

    record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

    features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                             cls_token=tokenizer.cls_token,
                                             sep_token=tokenizer.sep_token,
                                             cls_token_segment_id=1,
                                             pad_token_segment_id=0,
                                             disable_progress_bar=True)[0]

    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
        # for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
        for i, bert_input in list(enumerate(features)):
            logits[i] = model.ranking_linear(
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                )[1]
            )
        scores = softmax(logits, dim=0)

        preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))

    # print (preds)
    sense = preds[0][0]
    meaning = preds[0][1]
    return (sense, meaning, ambiguous_word)


def get_question(sentence, answer):
    text = "context: {} answer: {} </s>".format(sentence, answer)
    # print (text)
    max_len = 256
    encoding = question_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True, return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = question_model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   early_stopping=True,
                                   num_beams=5,
                                   num_return_sequences=1,
                                   no_repeat_ngram_size=2,
                                   max_length=200)

    dec = [question_tokenizer.decode(ids) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


def getMCQs(sent):
    sentence_for_bert = sent.replace("**", " [TGT] ")
    sentence_for_bert = " ".join(sentence_for_bert.split())
    # try:
    sense, meaning, answer = get_sense(sentence_for_bert)
    if sense is not None:
        distractors = get_distractors_wordnet(sense, answer)
    else:
        distractors = ["Word not found in Wordnet. So unable to extract distractors."]
    sentence_for_T5 = sent.replace("**", " ")
    sentence_for_T5 = " ".join(sentence_for_T5.split())
    ques = get_question(sentence_for_T5, answer)
    return ques, answer, distractors, meaning


def mcq_sent(sentence, target):
    temp = "**" + target + "**"
    return sentence.replace(target, temp)


def mcq_csv_generate(convert_to_mcq):
    question_list = []
    answer_list = []
    mcq_list = []

    for index, row in convert_to_mcq.iterrows():
        print(row["Sentences"], row["Target"])

        question, answer, distractors, meaning = getMCQs(mcq_sent(row["Sentences"].lower(), row["Target"].lower()))

        question_list.append(question)

        if len(distractors) < 2:
            mcq = ["NA"]
        else:
            mcq = distractors[:3]
            mcq.append(answer)
            random.shuffle(mcq)

        mcq_list.append(mcq)

        answer_list.append(answer)

    dict_ = {
        "Questions": question_list,
        "Options": mcq_list,
        "Answers": answer_list
    }

    return pd.DataFrame(dict_)
