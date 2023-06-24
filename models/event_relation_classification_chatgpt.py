import os, json
import backoff
import openai
###
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def convert_label(label, task):
    if task == "binary":
        if label in ['required argument', 'optional argument']:
            label = "argument"
    if label in ['independent', 'condition']:
        label = "not an argument"
    return label

def get_chat_input(events):
    return "EVENT 1: " + events['Event-Event span'].split("[SEP]")[0] + "; EVENT 2: " + events['Event-Event span'].split("[SEP]")[-1]

with open("./data/test.json", "r") as f:
    test_data = json.load(f)

test_data = test_data[:]
for data in range(len(test_data)):
    test_data[data]["chat_completion_input"] = get_chat_input(test_data[data])


task = "three"
input_type = "Marked-predicate sentence"# Marked-predicate sentence  Event-Event-SRL-DEP Event-Event-SRL  Event-Event span


messages = [{"role": "system", "content": "You’re helping classify the input into three classes: required argument, optional argument, and not an argument."}]


if input_type == "Event-Event span":
    first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family\n" \
             "class: required argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it\n" \
             "class: optional argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth\n" \
             "class: not an argument\n" \
             "###\n"
    if task == "binary":
        first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family\n" \
             "class: argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it\n" \
             "class: argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth\n" \
             "class: not an argument\n" \
             "###\n"
        messages = [{"role": "system", "content": "You’re helping classify the input into two classes: argument and not an argument."}]


if input_type == "Event-Event-SRL":
    first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family[SRL]ARG1\n" \
             "class: required argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it[SRL]ARG1\n" \
             "class: optional argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers[SRL]ARG1\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth[SRL]ARG1\n" \
             "class: not an argument\n" \
             "###\n"
    if task == "binary":
        first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family[SRL]ARG1\n" \
             "class: argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it[SRL]ARG1\n" \
             "class: argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers[SRL]ARG1\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth[SRL]ARG1\n" \
             "class: not an argument\n" \
             "###\n"
        messages = [{"role": "system", "content": "You’re helping classify the input into two classes: argument and not an argument."}]


if input_type == "Event-Event-SRL-DEP":
    first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family[SRL]ARG1[DEP]xcomp\n" \
             "class: required argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it[SRL]ARG1[DEP]xcomp\n" \
             "class: optional argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers[SRL]ARG1[DEP]parataxis\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth[SRL]ARG1[DEP]advcl\n" \
             "class: not an argument\n" \
             "###\n"
    if task == "binary":
        first_content = "events: Jenny {V: tries} to stop them from ruining her family[SEP]Jenny {V: stop} them from ruining her family[SRL]ARG1[DEP]xcomp\n" \
             "class: argument\n" \
             "###\n" \
             "events: Brodski {V: go} EVA to fix it[SEP]Brodski {V: fix} it[SRL]ARG1[DEP]xcomp\n" \
             "class: argument\n" \
             "###\n" \
             "events: It {V: increases} the discount rate it offers to known customers[SEP]the discount rate it {V: offers} to known customers[SRL]ARG1[DEP]parataxis\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will {V: hate} her if she tells the truth[SEP]she {V: tells} the truth[SRL]ARG1[DEP]advcl\n" \
             "class: not an argument\n" \
             "###\n"
        messages = [{"role": "system", "content": "You’re helping classify the input into two classes: argument and not an argument."}]

if input_type == "Marked-predicate sentence":
    first_content = "events: Jenny [V1] tries [\V1] to [V2] stop [\V2] them from ruining her family\n" \
             "class: required argument\n" \
             "###\n" \
             "events: Brodski [V1] go [\V1] EVA to [V2] fix [\V2] it\n" \
             "class: optional argument\n" \
             "###\n" \
             "events: It [V1] increases [\V1] the discount rate it [V2] offers [\V2] to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will [V1] hate [\V1] her if she [V2] tells [\V2] the truth\n" \
             "class: not an argument\n" \
             "###\n"
    if task == "binary":
        first_content = "events: Jenny [V1] tries [\V1] to [V2] stop [\V2] them from ruining her family\n" \
             "class: argument\n" \
             "###\n" \
             "events: Brodski [V1] go [\V1] EVA to [V2] fix [\V2] it\n" \
             "class: argument\n" \
             "###\n" \
             "events: It [V1] increases [\V1] the discount rate it [V2] offers [\V2] to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "events: Joe will [V1] hate [\V1] her if she [V2] tells [\V2] the truth\n" \
             "class: not an argument\n" \
             "###\n"
        messages = [{"role": "system", "content": "You’re helping classify the input into two classes: argument and not an argument."}]


if input_type == "chat_completion_input":
    messages = [{"role": "system", "content": "You’re helping classify the relation between Event 1 and Event 2 into three classes: required argument, optional argument, and not an argument."}]
    first_content = "EVENT 1: Jenny {V: tries} to stop them from ruining her family; EVENT 2: Jenny {V: stop} them from ruining her family\n" \
             "class: required argument\n" \
             "###\n" \
             "EVENT 1: Brodski {V: go} EVA to fix it; EVENT 2: Brodski {V: fix} it\n" \
             "class: optional argument\n" \
             "###\n" \
             "EVENT 1: It {V: increases} the discount rate it offers to known customers; EVENT 2: the discount rate it {V: offers} to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "EVENT 1: Joe will {V: hate} her if she tells the truth; EVENT 2: she {V: tells} the truth\n" \
             "class: not an argument\n" \
             "###\n"
    if task == "binary":
        first_content = "EVENT 1: Jenny {V: tries} to stop them from ruining her family; EVENT 2: Jenny {V: stop} them from ruining her family\n" \
             "class: argument\n" \
             "###\n" \
             "EVENT 1: Brodski {V: go} EVA to fix it; EVENT 2: Brodski {V: fix} it\n" \
             "class: argument\n" \
             "###\n" \
             "EVENT 1: It {V: increases} the discount rate it offers to known customers; EVENT 2: the discount rate it {V: offers} to known customers\n" \
             "class: not an argument\n" \
             "###\n" \
             "EVENT 1: Joe will {V: hate} her if she tells the truth; EVENT 2: she {V: tells} the truth\n" \
             "class: not an argument\n" \
             "###\n"
        messages = [{"role": "system", "content": "You’re helping classify the relation between Event 1 and Event 2 into two classes: argument and not an argument."}]




prediction = []
GT = []
messages.append("")
for i in range(len(test_data)):
    print("start ", i)

    content = first_content + "events: " + test_data[i][input_type] + "\n class: \n"

    print(f'content: {test_data[i][input_type]}')
    messages[1] = {"role": "user", "content": content}
    completion = completions_with_backoff(model="gpt-3.5-turbo",
                                          messages=messages)

    chat_response = completion.choices[0].message.content
    prediction.append(chat_response)
    GT.append(convert_label(test_data[i]["label"], task))
    print(f'ChatGPT: {chat_response}')
    print(f'true label: {convert_label(test_data[i]["label"], task)}\n')

print("prediction = ", prediction)
print("label = ", GT)
