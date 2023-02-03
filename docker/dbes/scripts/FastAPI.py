#!/usr/bin/env python

import logging

logerror = logging.error
loginfo = logging.info
from collections import defaultdict
from json import load

from fastapi import FastAPI, Request
from base64 import b64encode, b64decode

loginfo(f" | LOADING CONFIGS TO RAM...")
config_data = defaultdict()
path = r"/app/config.json"
with open(path) as json_file:
    try:
        config_data = load(json_file)
    except Exception as e:
        logerror("Cant read config from json file!", exc_info=True)


import sys

sys.path.append("/app/ES")
from dbes.DBESGraphVisualizer import DBESGraphVisualizer
from dbes.DBESNet import DBESNet
from dbes.DBESNode import DBESNode
from es.ES import ES

dbes = DBESNet()
es = ES()

# initialization
app = FastAPI()


async def _template(func, request):
    """
    #-USAGE------------------------------------
        | Used to receive messages on the set port (in main.py).
    #-ARGUMENTS--------------------------------
        | - request          - Message from POST. Task, that needed be executed.
    """
    try:
        task_msg = await request.json()
    except Exception as e:
        logerror("Cant read json data!", exc_info=True)
        return {"error": "Cant read json data!"}

    try:
        result_msg = {"error": "Error occured while execute task!"}
        loginfo(f" | TASK STARTED")

        result_msg = await func(task_msg)

        loginfo(f" | TASK DONE")
    except Exception as e:
        logerror(
            "Error occured while doing task in QUEUE! Task aborted!", exc_info=True
        )
        return {"error": "Error occured while doing task in QUEUE! Task aborted!"}

    return result_msg


################################################################################################################################
# # API DBESNet
################################################################################################################################
async def get_adjacency_matrix(task_msg):
    result_msg = dbes.get_adjacency_matrix(**task_msg)
    return result_msg


@app.post("/get_adjacency_matrix/")
async def get_adjacency_matrix_app(request: Request):
    result_msg = await _template(get_adjacency_matrix, request)
    return result_msg


async def get_adjacency_list(task_msg):
    result_msg = dbes.get_adjacency_list(**task_msg)
    return result_msg


@app.post("/get_adjacency_list/")
async def get_adjacency_list_app(request: Request):
    result_msg = await _template(get_adjacency_list, request)
    return result_msg


async def get_net(task_msg):
    result_msg = dbes.get_net(**task_msg)
    return result_msg


@app.post("/get_net/")
async def get_net_app(request: Request):
    result_msg = await _template(get_net, request)
    return result_msg


async def get_net_file(task_msg):
    result_msg = dbes.get_net_file(**task_msg)
    return result_msg


@app.post("/get_net_file/")
async def get_net_file_app(request: Request):
    result_msg = await _template(get_net_file, request)
    return result_msg


async def get_common_file(task_msg):
    result_msg = dbes.get_common_file(task_msg)
    return result_msg


@app.post("/get_common_file/")
async def get_common_file_app(request: Request):
    result_msg = await _template(get_common_file, request)
    return result_msg


async def get_templates(task_msg):
    result_msg = dbes.get_templates(**task_msg)
    return result_msg


@app.post("/get_templates/")
async def get_templates_app(request: Request):
    result_msg = await _template(get_templates, request)
    return result_msg


async def get_template_file(task_msg):
    result_msg = dbes.get_template_file(**task_msg)
    return result_msg


@app.post("/get_template_file/")
async def get_template_file_app(request: Request):
    result_msg = await _template(get_template_file, request)
    return result_msg


async def get_value_rel(task_msg):
    result_msg = dbes.get_value_rel(**task_msg)
    return result_msg


@app.post("/get_value_rel/")
async def get_value_rel_app(request: Request):
    result_msg = await _template(get_value_rel, request)
    return result_msg


async def get_sum_of_values(task_msg):
    result_msg = dbes.get_sum_of_values(**task_msg)
    return result_msg


@app.post("/get_sum_of_values/")
async def get_sum_of_values_app(request: Request):
    result_msg = await _template(get_sum_of_values, request)
    return result_msg


async def remove_net_file(task_msg):
    result_msg = dbes.remove_net_file(**task_msg)
    return result_msg


@app.post("/remove_net_file/")
async def remove_net_file_app(request: Request):
    result_msg = await _template(remove_net_file, request)
    return result_msg


async def remove_common_file(task_msg):
    result_msg = dbes.remove_common_file(**task_msg)
    return result_msg


@app.post("/remove_common_file/")
async def remove_common_file_app(request: Request):
    result_msg = await _template(remove_common_file, request)
    return result_msg


async def remove_template_file(task_msg):
    result_msg = dbes.remove_template_file(**task_msg)
    return result_msg


@app.post("/remove_template_file/")
async def remove_template_file_app(request: Request):
    result_msg = await _template(remove_template_file, request)
    return result_msg


async def remove_profile(task_msg):
    result_msg = dbes.remove_profile(**task_msg)
    return result_msg


@app.post("/remove_profile/")
async def remove_profile_app(request: Request):
    result_msg = await _template(remove_profile, request)
    return result_msg


async def remove_node_idx(task_msg):
    result_msg = dbes.remove_node_idx(**task_msg)
    return result_msg


@app.post("/remove_node_idx/")
async def remove_node_idx_app(request: Request):
    result_msg = await _template(remove_node_idx, request)
    return result_msg


async def remove_rel_between(task_msg):
    result_msg = dbes.remove_rel_between(**task_msg)
    return result_msg


@app.post("/remove_rel_between/")
async def remove_rel_between_app(request: Request):
    result_msg = await _template(remove_rel_between, request)
    return result_msg


async def add_profile_DBES(task_msg):
    result_msg = dbes.add_profile(**task_msg)
    return result_msg


@app.post("/add_profile_DBES/")
async def add_profile_DBES_app(request: Request):
    result_msg = await _template(add_profile_DBES, request)
    return result_msg


async def add_template(task_msg):
    result_msg = dbes.add_template(**task_msg)
    return result_msg


@app.post("/add_template/")
async def add_template_app(request: Request):
    result_msg = await _template(add_template, request)
    return result_msg


async def add_common_file(task_msg):
    result_msg = dbes.add_common_file(**task_msg)
    return result_msg


@app.post("/add_common_file/")
async def add_common_file_app(request: Request):
    result_msg = await _template(add_common_file, request)
    return result_msg


async def add_template_file(task_msg):
    result_msg = dbes.add_template_file(**task_msg)
    return result_msg


@app.post("/add_template_file/")
async def add_template_file_app(request: Request):
    result_msg = await _template(add_template_file, request)
    return result_msg


async def export_to_zip(task_msg):
    path_save = dbes.export_to_zip()
    return b64encode(open(path_save, "rb").read()).decode("utf-8")


@app.post("/export_to_zip/")
async def export_to_zip_app(request: Request):
    result_msg = await _template(export_to_zip, request)
    return result_msg


async def import_from_zip(task_msg):
    path_save = dbes._CONFIG["save"]["path_save"][:-4] + "import.zip"
    with open(path_save, "wb") as f:
        f.write(b64decode(task_msg.encode("utf-8")))
    dbes.import_from_zip(path_save)
    return True


@app.post("/import_from_zip/")
async def import_from_zip_app(request: Request):
    result_msg = await _template(import_from_zip, request)
    return result_msg


async def import_from_sep_json(task_msg):
    result_msg = dbes.import_from_sep_json(**task_msg)
    return result_msg


@app.post("/import_from_sep_json/")
async def import_from_sep_json_app(request: Request):
    result_msg = await _template(import_from_sep_json, request)
    return result_msg


async def import_from_json(task_msg):
    result_msg = dbes.import_from_json(**task_msg)
    return result_msg


@app.post("/import_from_json/")
async def import_from_json_app(request: Request):
    result_msg = await _template(import_from_json, request)
    return result_msg


async def import_from_bad_json(task_msg):
    result_msg = dbes.import_from_bad_json(task_msg)
    return result_msg


@app.post("/import_from_bad_json/")
async def import_from_bad_json_app(request: Request):
    result_msg = await _template(import_from_bad_json, request)
    return result_msg


async def import_from_sep_table(task_msg):
    result_msg = dbes.import_from_sep_table(**task_msg)
    return result_msg


@app.post("/import_from_sep_table/")
async def import_from_sep_table_app(request: Request):
    result_msg = await _template(import_from_sep_table, request)
    return result_msg


async def import_from_sep_csv(task_msg):
    result_msg = dbes.import_from_sep_csv(**task_msg)
    return result_msg


@app.post("/import_from_sep_csv/")
async def import_from_sep_csv_app(request: Request):
    result_msg = await _template(import_from_sep_csv, request)
    return result_msg


async def import_from_postgres(task_msg):
    result_msg = dbes.import_from_postgres(task_msg)
    return result_msg


@app.post("/import_from_postgres/")
async def import_from_postgres_app(request: Request):
    result_msg = await _template(import_from_postgres, request)
    return result_msg


async def find_template_val_idxs(task_msg):
    result_msg = dbes.find_template_val_idxs(**task_msg)
    return result_msg


@app.post("/find_template_val_idxs/")
async def find_template_val_idxs_app(request: Request):
    result_msg = await _template(find_template_val_idxs, request)
    return result_msg


async def find_template_rel_idxs(task_msg):
    result_msg = dbes.find_template_rel_idxs(**task_msg)
    return result_msg


@app.post("/find_template_rel_idxs/")
async def find_template_rel_idxs_app(request: Request):
    result_msg = await _template(find_template_rel_idxs, request)
    return result_msg


async def find_template_mult_rel_idxs(task_msg):
    result_msg = dbes.find_template_mult_rel_idxs(**task_msg)
    return result_msg


@app.post("/find_template_mult_rel_idxs/")
async def find_template_mult_rel_idxs_app(request: Request):
    result_msg = await _template(find_template_mult_rel_idxs, request)
    return result_msg


async def find_all_out_idxs(task_msg):
    result_msg = dbes.find_all_out_idxs(**task_msg)
    return result_msg


@app.post("/find_all_out_idxs/")
async def find_all_out_idxs_app(request: Request):
    result_msg = await _template(find_all_out_idxs, request)
    return result_msg


async def find_all_in_idxs(task_msg):
    result_msg = dbes.find_all_in_idxs(**task_msg)
    return result_msg


@app.post("/find_all_in_idxs/")
async def find_all_in_idxs_app(request: Request):
    result_msg = await _template(find_all_in_idxs, request)
    return result_msg


async def find_all_dir_rel_idxs_lvls(task_msg):
    result_msg = dbes.find_all_dir_rel_idxs_lvls(**task_msg)
    return result_msg


@app.post("/find_all_dir_rel_idxs_lvls/")
async def find_all_dir_rel_idxs_lvls_app(request: Request):
    result_msg = await _template(find_all_dir_rel_idxs_lvls, request)
    return result_msg


async def find_type_rel_idxs(task_msg):
    result_msg = dbes.find_type_rel_idxs(**task_msg)
    return result_msg


@app.post("/find_type_rel_idxs/")
async def find_type_rel_idxs_app(request: Request):
    result_msg = await _template(find_type_rel_idxs, request)
    return result_msg


async def find_rel_idxs(task_msg):
    result_msg = dbes.find_rel_idxs_NORec(**task_msg)
    return result_msg


@app.post("/find_rel_idxs/")
async def find_rel_idxs_app(request: Request):
    result_msg = await _template(find_rel_idxs, request)
    return result_msg


async def find_mult_rel_idxs(task_msg):
    result_msg = dbes.find_mult_rel_idxs(**task_msg)
    return result_msg


@app.post("/find_mult_rel_idxs/")
async def find_mult_rel_idxs_app(request: Request):
    result_msg = await _template(find_mult_rel_idxs, request)
    return result_msg


async def find_val_idxs(task_msg):
    result_msg = dbes.find_val_idxs(**task_msg)
    return result_msg


@app.post("/find_val_idxs/")
async def find_val_idxs_app(request: Request):
    result_msg = await _template(find_val_idxs, request)
    return result_msg


async def find_idxs_type_rel_coeff(task_msg):
    result_msg = dbes.find_idxs_type_rel_coeff(**task_msg)
    return result_msg


@app.post("/find_idxs_type_rel_coeff/")
async def find_idxs_type_rel_coeff_app(request: Request):
    result_msg = await _template(find_idxs_type_rel_coeff, request)
    return result_msg


async def find_and_combine_idxs_type_rel_coeff(task_msg):
    result_msg = dbes.find_and_combine_idxs_type_rel_coeff(**task_msg)
    return result_msg


@app.post("/find_and_combine_idxs_type_rel_coeff/")
async def find_and_combine_idxs_type_rel_coeff_app(request: Request):
    result_msg = await _template(find_and_combine_idxs_type_rel_coeff, request)
    return result_msg


async def find_idxs_val_rel_coeff(task_msg):
    result_msg = dbes.find_idxs_val_rel_coeff(**task_msg)
    return result_msg


@app.post("/find_idxs_val_rel_coeff/")
async def find_idxs_val_rel_coeff_app(request: Request):
    result_msg = await _template(find_idxs_val_rel_coeff, request)
    return result_msg


async def find_and_combine_idxs_val_rel(task_msg):
    result_msg = dbes.find_and_combine_idxs_val_rel(**task_msg)
    return result_msg


@app.post("/find_and_combine_idxs_val_rel/")
async def find_and_combine_idxs_val_rel_app(request: Request):
    result_msg = await _template(find_and_combine_idxs_val_rel, request)
    return result_msg


async def find_and_union_idxs_val_rel(task_msg):
    result_msg = dbes.find_and_union_idxs_val_rel(**task_msg)
    return result_msg


@app.post("/find_and_union_idxs_val_rel/")
async def find_and_union_idxs_val_rel_app(request: Request):
    result_msg = await _template(find_and_union_idxs_val_rel, request)
    return result_msg


async def find_back_pattern_idxs(task_msg):
    result_msg = dbes.find_back_pattern_idxs(**task_msg)
    return result_msg


@app.post("/find_back_pattern_idxs/")
async def find_back_pattern_idxs_app(request: Request):
    result_msg = await _template(find_back_pattern_idxs, request)
    return result_msg


async def find_pattern_idxs(task_msg):
    result_msg = dbes.find_pattern_idxs(**task_msg)
    return result_msg


@app.post("/find_pattern_idxs/")
async def find_pattern_idxs_app(request: Request):
    result_msg = await _template(find_pattern_idxs, request)
    return result_msg


async def find_structure_idxs(task_msg):
    result_msg = dbes.find_structure_idxs(**task_msg)
    return result_msg


@app.post("/find_structure_idxs/")
async def find_structure_idxs_app(request: Request):
    result_msg = await _template(find_structure_idxs, request)
    return result_msg


async def find_shortest_dist_z(task_msg):
    result_msg = dbes.find_shortest_dist_z(**task_msg)
    return result_msg


@app.post("/find_shortest_dist_z/")
async def find_shortest_dist_z_app(request: Request):
    result_msg = await _template(find_shortest_dist_z, request)
    return result_msg


async def find_shortest_dist_dijkstra_amx(task_msg):
    result_msg = dbes.find_shortest_dist_dijkstra_amx(**task_msg)
    return result_msg


@app.post("/find_shortest_dist_dijkstra_amx/")
async def find_shortest_dist_dijkstra_amx_app(request: Request):
    result_msg = await _template(find_shortest_dist_dijkstra_amx, request)
    return result_msg


async def find_shortest_dist_dijkstra_als(task_msg):
    result_msg = dbes.find_shortest_dist_dijkstra_als(**task_msg)
    return result_msg


@app.post("/find_shortest_dist_dijkstra_als/")
async def find_shortest_dist_dijkstra_als_app(request: Request):
    result_msg = await _template(find_shortest_dist_dijkstra_als, request)
    return result_msg


async def find_shortest_dist_DAG(task_msg):
    result_msg = dbes.find_shortest_dist_DAG(**task_msg)
    return result_msg


@app.post("/find_shortest_dist_DAG/")
async def find_shortest_dist_DAG_app(request: Request):
    result_msg = await _template(find_shortest_dist_DAG, request)
    return result_msg


async def find_shortest_dist_fw(task_msg):
    result_msg = dbes.find_shortest_dist_fw(**task_msg)
    return result_msg


@app.post("/find_shortest_dist_fw/")
async def find_shortest_dist_fw_app(request: Request):
    result_msg = await _template(find_shortest_dist_fw, request)
    return result_msg


async def find_longest_mult_dist_fw(task_msg):
    result_msg = dbes.find_longest_mult_dist_fw(**task_msg)
    return result_msg


@app.post("/find_longest_mult_dist_fw/")
async def find_longest_mult_dist_fw_app(request: Request):
    result_msg = await _template(find_longest_mult_dist_fw, request)
    return result_msg


async def find_from_template(task_msg):
    result_msg = dbes.find_from_template(**task_msg)
    return result_msg


@app.post("/find_from_template/")
async def find_from_template_app(request: Request):
    result_msg = await _template(find_from_template, request)
    return result_msg


async def find(task_msg):
    result_msg = dbes.find(task_msg)
    return result_msg


@app.post("/find/")
async def find_app(request: Request):
    result_msg = await _template(find, request)
    return result_msg


async def sort_idxs(task_msg):
    result_msg = dbes.sort(**task_msg)
    return result_msg


@app.post("/sort/")
async def sort_idxs_app(request: Request):
    result_msg = await _template(sort_idxs, request)
    return result_msg


################################################################################################################################


################################################################################################################################
# # API ES
################################################################################################################################
async def probq_one_lvl_one_target(task_msg):
    result_msg = es.probq_one_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/probq_one_lvl_one_target/")
async def probq_one_lvl_one_target_app(request: Request):
    result_msg = await _template(probq_one_lvl_one_target, request)
    return result_msg


async def probq_MHA_lvl_one_target(task_msg):
    result_msg = es.probq_MHA_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/probq_MHA_lvl_one_target/")
async def probq_MHA_lvl_one_target_app(request: Request):
    result_msg = await _template(probq_MHA_lvl_one_target, request)
    return result_msg


async def probq_MnHA_lvl_one_target(task_msg):
    result_msg = es.probq_MnHA_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/probq_MnHA_lvl_one_target/")
async def probq_MnHA_lvl_one_target_app(request: Request):
    result_msg = await _template(probq_MnHA_lvl_one_target, request)
    return result_msg


async def text_find_one_lvl_one_target(task_msg):
    result_msg = es.text_find_one_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/text_find_one_lvl_one_target/")
async def text_find_one_lvl_one_target_app(request: Request):
    result_msg = await _template(text_find_one_lvl_one_target, request)
    return result_msg


async def text_find_MHA_lvl_one_target(task_msg):
    result_msg = es.text_find_MHA_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/text_find_MHA_lvl_one_target/")
async def text_find_MHA_lvl_one_target_app(request: Request):
    result_msg = await _template(text_find_MHA_lvl_one_target, request)
    return result_msg


async def text_find_MnHA_lvl_one_target(task_msg):
    result_msg = es.text_find_MnHA_lvl_one_target(dbes=dbes, **task_msg)
    return result_msg


@app.post("/text_find_MnHA_lvl_one_target/")
async def text_find_MnHA_lvl_one_target_app(request: Request):
    result_msg = await _template(text_find_MnHA_lvl_one_target, request)
    return result_msg


async def add_profile_ES(task_msg):
    result_msg = es.add_profile(dbes=dbes, **task_msg)
    return result_msg


@app.post("/add_profile_ES/")
async def add_profile_ES_app(request: Request):
    result_msg = await _template(add_profile_ES, request)
    return result_msg


################################################################################################################################


################################################################################################################################
# # API DBES Graph Visualization
################################################################################################################################
async def GV_visualize(task_msg):
    dbes_gv = DBESGraphVisualizer(
        {idx: dbes._net[idx] for idx in task_msg["nodes_idxs"]}
    )
    return dbes_gv.visualize()


@app.post("/GV_visualize/")
async def GV_visualize_app(request: Request):
    result_msg = await _template(GV_visualize, request)
    return result_msg


################################################################################################################################
