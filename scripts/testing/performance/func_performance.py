import time
from pprint import pprint

import sys

sys.path.insert(0, "scripts")

from dbes.DBESNet import DBESNet
from dbes.DBESNode import DBESNode
from es.ES import ES

st = time.time()
d = DBESNet()
es = ES()
print("Time up: ", time.time() - st)

################################################################################################################################
# Тест производительности функций
################################################################################################################################
for i in range(10):
    print("--------------------------------------")
    st = time.time()
    important_nodes = {}
    important_nodes["Словарь"] = d.find_val_idxs("Словарь").pop()
    important_nodes["id"] = d.find_val_idxs("id").pop()
    important_nodes["eid"] = d.find_val_idxs("eid").pop()
    important_nodes["Услуга"] = d.find_val_idxs("Услуга").pop()
    important_nodes["Тег услуги"] = d.find_val_idxs("Тег услуги").pop()
    important_nodes["Параметр тега"] = d.find_val_idxs("Параметр тега").pop()
    important_nodes["Параметр услуги"] = d.find_val_idxs("Параметр услуги").pop()
    important_nodes["SERVICE"] = d.find_val_idxs("SERVICE").pop()
    important_nodes["CONSULTATION"] = d.find_val_idxs("CONSULTATION").pop()
    important_nodes["DEPERSONALIZED_CONSULTATION"] = d.find_val_idxs(
        "DEPERSONALIZED_CONSULTATION"
    ).pop()
    important_nodes["Портал госуслуг"] = d.find_val_idxs("Портал госуслуг").pop()
    print("Time find_val_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_all_out_idxs(important_nodes["SERVICE"])
    print("Time find_all_out_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_type_rel_idxs("prob", "out")
    print("Time find_type_rel_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_rel_idxs(important_nodes["Услуга"])
    print("Time find_rel_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_rel_idxs(important_nodes["eid"], recursive=True)
    print("Time find_rel_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_rel_idxs_NORec(important_nodes["eid"], recursive=True)
    print("Time find_rel_idxs_NORec: ", (time.time() - st))

    st = time.time()
    a = d.find_mult_rel_idxs(
        {"in": important_nodes["id"], "out": important_nodes["Портал госуслуг"]}
    )
    print("Time find_mult_rel_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_val_idxs("а", contains=True)
    print("Time find_val_idxs: ", (time.time() - st))

    st = time.time()
    a = d.find_rel_idxs_NORec(important_nodes["Услуга"], recursive=True)
    print("Time find_rel_idxs_NORec: ", (time.time() - st))

    fi = d.find_rel_idxs_NORec(important_nodes["Параметр тега"])
    tag_params = {d._net[idx].value: idx for idx in fi}

    start_time = time.time()
    answer = es.probq_MHA_lvl_one_target(
        target_rel_idxs={"in": [important_nodes["Услуга"]]},
        question_rel_idxs={"in": [important_nodes["Тег услуги"]]},
        answers={
            d.find_val_idxs("Портал госуслуг").pop(): "on",
            d.find_val_idxs("У заявителя есть Подтвержденная УЗ").pop(): "on",
        },
        true_stats_idxs={"in": [important_nodes["SERVICE"]]},
        all_stats_idxs={
            "in": [
                important_nodes["SERVICE"],
                important_nodes["CONSULTATION"],
                important_nodes["DEPERSONALIZED_CONSULTATION"],
            ]
        },
        question_add_props_idxs={
            "question": tag_params["Вопрос"],
            "description": tag_params["Описание"],
        },
        target_add_props_idxs={"eid": important_nodes["eid"]},
        top_questions_output=5,
        top_targets_output=5,
        dbes=d,
    )
    print("Time probq_MHA_lvl_one_target: ", (time.time() - start_time))  # 0.0534

    start_time = time.time()
    answer = es.probq_MHA_lvl_one_target(
        answers={
            d.find_val_idxs("Портал госуслуг").pop(): "on",
            d.find_val_idxs("У заявителя есть Подтвержденная УЗ").pop(): "on",
        },
        dbes=d,
        _profile="Опросник Иркутска",
        _runtype="args",
    )
    print("Time probq_MHA_lvl_one_target(caching): ", (time.time() - start_time))

    func_args = {
        "target_rel_idxs": {"in": [important_nodes["Тег услуги"]]},
        "target_stats_idxs": {"in": [important_nodes["SERVICE"]]},
        "target_rule_sort": "value_recursive_relation",
        "target_add_text_idxs": {
            "question": tag_params["Вопрос"],
            "description": tag_params["Описание"],
        },
        "vocab_rel_idxs": {"in": important_nodes["Словарь"]},
    }
    start_time = time.time()
    answer = es.text_find_one_lvl_one_target(input_text="родня", dbes=d, **func_args)
    print("Time text_find_one_lvl_one_target: ", (time.time() - start_time))

    start_time = time.time()
    tags = es.text_find_one_lvl_one_target(
        input_text="родня",
        dbes=d,
        _profile="Поисковая строка Иркутска",
        _runtype="args",
    )
    print("Time text_find_one_lvl_one_target(caching): ", (time.time() - start_time))

    start_time = time.time()
    tags = es.text_find_MHA_lvl_one_target(
        input_text="родня",
        dbes=d,
        _profile="Поисковая строка Иркутска",
        _runtype="args",
    )
    print("Time text_find_MHA_lvl_one_target(caching): ", (time.time() - start_time))

    start_time = time.time()
    tags = es.text_find_MnHA_lvl_one_target(
        input_text="родня",
        # type_output='short',
        dbes=d,
        _profile="Поисковая строка Иркутска",
        _runtype="args",
    )
    print("Time text_find_MnHA_lvl_one_target(caching): ", (time.time() - start_time))

"""
Python 3.7.10
Time find_val_idxs:  0.01959514617919922
Time find_all_out_idxs:  0.0017864704132080078
Time find_type_rel_idxs:  0.009652137756347656
Time find_rel_idxs:  0.004479169845581055
Time find_rel_idxs:  0.021939754486083984
Time find_rel_idxs_NORec:  0.007077217102050781
Time find_mult_rel_idxs:  0.006838321685791016
Time find_val_idxs:  0.006936788558959961
Time find_rel_idxs_NORec:  0.004907131195068359
Time probq_MHA_lvl_one_target:  0.05755114555358887
Time probq_MHA_lvl_one_target(caching):  0.006598711013793945
Time text_find_one_lvl_one_target:  0.45023083686828613
Time text_find_one_lvl_one_target:  0.08664774894714355 ? caching estf?
Time text_find_one_lvl_one_target nltk tokenizer:  0.4679131507873535
Time text_find_one_lvl_one_target(caching):  0.0003902912139892578
Time text_find_MHA_lvl_one_target(caching):  0.0011425018310546875
Time text_find_MnHA_lvl_one_target(caching):  0.001232147216796875

Python 3.8.8
Time find_val_idxs:  0.019509553909301758
Time find_all_out_idxs:  0.0016400814056396484
Time find_type_rel_idxs:  0.010165214538574219
Time find_rel_idxs:  0.007474184036254883
Time find_rel_idxs:  0.02187180519104004
Time find_rel_idxs_NORec:  0.007668733596801758
Time find_mult_rel_idxs:  0.007691383361816406
Time find_val_idxs:  0.006561994552612305
Time find_rel_idxs_NORec:  0.00600433349609375
Time probq_MHA_lvl_one_target:  0.05816173553466797

Python 3.9.7
Time find_val_idxs:  0.024505138397216797
Time find_all_out_idxs:  0.0017745494842529297
Time find_type_rel_idxs:  0.011015176773071289
Time find_rel_idxs:  0.006737232208251953
Time find_rel_idxs:  0.023198366165161133
Time find_rel_idxs_NORec:  0.009495258331298828
Time find_mult_rel_idxs:  0.008576154708862305
Time find_val_idxs:  0.007220029830932617
Time find_rel_idxs_NORec:  0.005921602249145508
Time probq_MHA_lvl_one_target:  0.06002354621887207

Python 3.10.0
Time find_val_idxs:  0.01814723014831543
Time find_all_out_idxs:  0.0015041828155517578
Time find_type_rel_idxs:  0.01035308837890625
Time find_rel_idxs:  0.005677223205566406
Time find_rel_idxs:  0.021517038345336914
Time find_rel_idxs_NORec:  0.007416486740112305
Time find_mult_rel_idxs:  0.007928133010864258
Time find_val_idxs:  0.006087303161621094
Time find_rel_idxs_NORec:  0.0050356388092041016
Time probq_MHA_lvl_one_target:  0.05659961700439453

PyPy 7.3.7 with GCC 9.4.0
Time find_val_idxs:  0.007658481597900391
Time find_all_out_idxs:  0.0029630661010742188
Time find_type_rel_idxs:  0.005698204040527344
Time find_rel_idxs:  0.003810882568359375
Time find_rel_idxs:  0.015388011932373047
Time find_rel_idxs_NORec:  0.003701925277709961
Time find_mult_rel_idxs:  0.004381656646728516
Time find_val_idxs:  0.002482175827026367
Time find_rel_idxs_NORec:  0.003381967544555664
Time probq_MHA_lvl_one_target:  0.03806185722351074
Time probq_MHA_lvl_one_target(caching):  0.004004716873168945
Time text_find_one_lvl_one_target:  0.1209874153137207
Time text_find_one_lvl_one_target(caching):  0.0004076957702636719
Time text_find_MHA_lvl_one_target(caching):  0.0031709671020507812
Time text_find_MnHA_lvl_one_target(caching):  0.0051136016845703125

cython 0.29.14
Time find_val_idxs:  0.010763168334960938
Time find_all_out_idxs:  0.0012552738189697266
Time find_type_rel_idxs:  0.008637428283691406
Time find_rel_idxs:  0.004355669021606445
Time find_rel_idxs:  0.0159912109375
Time find_rel_idxs_NORec:  0.005803346633911133
Time find_mult_rel_idxs:  0.00508427619934082
Time find_val_idxs:  0.0033702850341796875
Time find_rel_idxs_NORec:  0.0033316612243652344
Time probq_MHA_lvl_one_target:  0.03733372688293457
Time probq_MHA_lvl_one_target(caching):  0.0011065006256103516

Pyston 2.3.1, GCC 9.4.0 Python 3.8.12
Time find_val_idxs:  0.01184535026550293
Time find_all_out_idxs:  0.0013175010681152344
Time find_type_rel_idxs:  0.00626683235168457
Time find_rel_idxs:  0.0036771297454833984
Time find_rel_idxs:  0.014021158218383789
Time find_rel_idxs_NORec:  0.00449681282043457
Time find_mult_rel_idxs:  0.004251241683959961
Time find_val_idxs:  0.0033762454986572266
Time find_rel_idxs_NORec:  0.003812074661254883
Time probq_MHA_lvl_one_target:  0.041773319244384766
Time probq_MHA_lvl_one_target(caching):  0.003923892974853516
Time text_find_one_lvl_one_target:  0.0725100040435791
Time text_find_one_lvl_one_target(caching):  0.00039076805114746094
Time text_find_MHA_lvl_one_target(caching):  0.0010800361633300781
Time text_find_MnHA_lvl_one_target(caching):  0.0013380050659179688

rust 1.57 bench
Time find_val_idxs:  0.0040886
Time find_all_out_idxs:  0.0001378
Time find_type_rel_idxs:  0.0039286
Time find_rel_idxs:  0.0021437
Time find_rel_idxs:  0.0074704
Time find_rel_idxs_NORec:  0.0011810
Time find_mult_rel_idxs:  0.00074875
Time find_val_idxs:  0.0015613
Time find_rel_idxs_NORec:  0.00096036


"""
