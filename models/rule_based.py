import pandas as pd
from collections import Counter, defaultdict
from pycorenlp import StanfordCoreNLP
from sklearn.metrics import classification_report, roc_auc_score
from utils import *
nlp = StanfordCoreNLP('http://localhost:9000')


def depenency_based_rule(sent_dep):
    ####get all verb index for all events  - all_verb_id_index as {event_id: verb_index}
    matched_pairs = []

    for dep in sent_dep:
        if dep[0][-4:] == "comp" or dep[0] == "csubj":
                matched_pairs.append((dep[1], dep[2]))
                # matched_events.append(["comp/csubj", [i["description"] for i in [df["event"]] + df["ACE"] if i["event_id"] == core_arg_pair[0]][0], [i["description"] for i in [df["event"]] + df["ACE"] if i["event_id"] == core_arg_pair[1]][0]])

        if dep[0] == "cop" and (dep[2], dep[1]) not in matched_pairs:
                matched_pairs.append((dep[2], dep[1]))
    W_word_list = ["so", "by", "about", "from", "in"]
    advcl_set = [dep for dep in sent_dep if dep[0] == "advcl"]
    mark_set = [dep for dep in sent_dep if dep[0] in ["mark", "case"]]
    if advcl_set and mark_set:
        flag = 0

        for pair in advcl_set:
        #### check if event1 and event2 verb are with advcl dep relation
        # if [v_1, v_2] in advcl_set or [v_2, v_1] in advcl_set:
            #### check if v_2 has mark dep relation with any other word
            for mark in mark_set:
                if pair[-1][0] in mark[-1]:
                    if pair[-1][0] == mark[-1][0]:
                        flag = mark[2]
                    else:
                        flag = mark[1]
                    if flag != 0 and flag.lower() in W_word_list and (dep[1], dep[2]) not in matched_pairs:
                        matched_pairs.append((dep[1], dep[2]))
                elif pair[-1][1] in mark[-1]:
                    if pair[-1][1] == mark[-1][0]:
                        flag = mark[2]
                    else:
                        flag = mark[1]
                    if flag != 0 and flag.lower() in W_word_list and (dep[1], dep[2]) not in matched_pairs:
                            matched_pairs.append((dep[1], dep[2]))

    return matched_pairs


def get_all_possible_sent(sent):

    dep = nlp.annotate(sent.replace("%", ""),
        properties={
            'annotators': 'depparse',
            'tokenize.language': 'Whitespace',
            'outputFormat': 'json',
            'timeout': 10000,
        })
    word_len = []
    dependencyRel_basic = []
    if len(dep["sentences"]) > 0:
        for sent in dep["sentences"]:
            word_len.append(len(sent["tokens"]))
            for dep_pair in sent["basicDependencies"]:
                if "punct" not in dep_pair["dep"]:
                    if len(word_len) > 1:
                        dependencyRel_basic.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"] + sum(word_len[:-1]), dep_pair["dependent"] + sum(word_len[:-1])]])
                    else:
                        dependencyRel_basic.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"], dep_pair["dependent"]]])

    else:
        # dependencyRel = 0
        dependencyRel_basic = 0

    # with open("/home/users/u7068796/ruiqi/story_planning/data/MovieSummaries/dep.pkl", 'wb') as f:
    #         pickle.dump(sent_pos_dep_dict, f)
    return dependencyRel_basic

def check_if_arg_event(df):
    if type(df["sent_split"]) is str:
        df["sent_split"] = eval(df["sent_split"])
    v1 = df["sent_split"][int(df["e1_verb_span"])]
    v2 = df["sent_split"][int(df["e2_verb_span"])]
    sent = " ".join(df["sent_split"] )
    matched_pairs = depenency_based_rule(get_all_possible_sent(sent))
    if (v1, v2) in matched_pairs:
        return 1
    else:
        return 0




if __name__ == '__main__':
    data_path = "../data/"
    selected_df = pd.DataFrame.from_records(load_json(data_path + "test.json"))
    numeric_label_dict = {"independent": 0, "condition": 0, "optional argument": 1, "required argument": 1}
    selected_df['bi_label'] = selected_df['label'].replace(numeric_label_dict)
    selected_df["pred_rule_based"] = selected_df.apply(lambda x: check_if_arg_event(x), axis=1)
    labels = selected_df['bi_label'].tolist()
    preds = selected_df["pred_rule_based"].tolist()
    print(classification_report(labels, preds, digits=4))
    print(roc_auc_score(labels, preds))







