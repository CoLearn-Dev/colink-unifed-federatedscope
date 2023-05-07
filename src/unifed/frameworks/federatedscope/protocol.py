import os
import json
import sys
import glob
import copy
import subprocess
import tempfile
from typing import List
import numpy as np

import colink as CL
import flbenchmark.datasets

from unifed.frameworks.federatedscope.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def convert_config(role, server_ip, client_ip):
    config = json.load(open('config.json', 'r'))
    # load dataset
    flbd = flbenchmark.datasets.FLBDatasets('../data')
    val_dataset = None
    if config['dataset'] == 'reddit':
        train_dataset, test_dataset, val_dataset = flbd.leafDatasets(config['dataset'])
    elif config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
        train_dataset, test_dataset = flbd.leafDatasets(config['dataset'])
    else:
        train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
    train_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_train')
    test_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_test')
    val_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_val')
    flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
    if test_dataset is not None:
        flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)
    if val_dataset is not None:
        flbenchmark.datasets.convert_to_csv(val_dataset, out_dir=val_data_base)

    client_num = len(glob.glob(f'../csv_data/{config["dataset"]}_train/*.csv'))

    if config['dataset'] == 'femnist':
        num_class = 62
        input_len = (28, 28)
        inplanes = 1
    elif config['dataset'] == 'celeba':
        num_class = 2
        input_len = (218, 178)
        inplanes = 3
    elif config['dataset'] == 'breast_horizontal':
        num_class = 2
        input_len = 30
        inplanes = 0
    elif config['dataset'] == 'default_credit_horizontal':
        num_class = 2
        input_len = 23
        inplanes = 0
    elif config['dataset'] == 'give_credit_horizontal':
        num_class = 2
        input_len = 10
        inplanes = 0
    elif config['dataset'] == 'student_horizontal':
        num_class = 1
        input_len = 13
        inplanes = 0
    elif config['dataset'] == 'vehicle_scale_horizontal':
        num_class = 4
        input_len = 18
        inplanes = 0
    elif config['dataset'].split('_')[-1] == 'vertical':
        loss = 'MSELoss' if config['dataset'] == 'dvisit_vertical' or config['dataset'] == 'motor_vertical' or config['dataset'] == 'student_vertical' else 'CrossEntropyLoss'
        metric = 'mse' if loss == 'MSELoss' else ('acc' if config['dataset'] == 'vehicle_scale_vertical' else 'roc_auc')
        vertical_config = f"""use_gpu: False
federate:
    mode: standalone
    client_num: 2
    total_round_num: {config['training_param']['epochs']}
    data_weighted_aggr: True
model:
    type: lr
    use_bias: True
train:
    optimizer:
        type: SGD
        lr: {config['training_param']['learning_rate']}
        momentum: {config['training_param']['optimizer_param']['momentum']}
data:
    type: synthetic_vfl_data
    batch_size: {config['training_param']['batch_size']}
vertical:
    use: True
    key_size: 256
trainer:
    type: none
criterion:
    type: {loss}
eval:
    freq: {int(config['training_param']['epochs']) + 10}
    metrics: ['{metric}']
    best_res_update_round_wise_key: test_loss
seed: {np.random.randint(2023)}
"""
        with open('./federatedscope/federatedscope/contrib/configs/vertical.yaml', 'w') as f:
            f.write(vertical_config)
        sys.exit()
    else:
        raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

    loss = 'MSELoss' if config['dataset'] == 'student_horizontal' else 'CrossEntropyLoss'
    metric = 'mse' if config['dataset'] == 'student_horizontal' else ('acc' if config['dataset'] == 'femnist' or config['dataset'] == 'celeba' or config['dataset'] == 'vehicle_scale_horizontal' else 'roc_auc')

    if config['model'].startswith('mlp_'):
        if config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
            input_len = inplanes * input_len[0] * input_len[1]
        sp = config['model'].split('_')
        model_type = 'mlp'
        if len(sp) < 2 or len(sp) > 4:
            raise NotImplementedError('Model {} is not supported.'.format(config['model']))
        hidden_layer = f'hidden: {sp[1]}'
        layer_num = f'layer: {len(sp)}'
    elif config['model'] == 'linear_regression' or config['model'] == 'logistic_regression':
        if config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
            input_len = inplanes * input_len[0] * input_len[1]
        sp = None
        model_type = 'lr' if config['model'] == 'logistic_regression' else 'mlp'
        hidden_layer = ''
        layer_num = f'layer: 1'
    elif config['model'] == 'lenet':
        if config['dataset'] != 'femnist' and config['dataset'] != 'celeba':
            raise NotImplementedError('Dataset {} is not supported for {}.'.format(config['dataset'], config['model']))
        sp = None
        model_type = 'mynet'
        hidden_layer = ''
        layer_num = ''
    else:
        raise NotImplementedError('Model {} is not supported.'.format(config['model']))

    if isinstance(input_len, int):
        input_len = tuple([input_len])

    server_config = f"""use_gpu: False
federate:
    client_num: {client_num}
    sample_client_num: {config['training_param']['client_per_round']}
    mode: 'distributed'
    total_round_num: {config['training_param']['epochs']}
    make_global_eval: False
    online_aggr: False
    data_weighted_aggr: True
distribute:
    use: True
    server_host: '{server_ip}'
    server_port: 50051
    role: 'server'
trainer:
    type: 'general'
train:
    optimizer:
        type: SGD
        lr: {config['training_param']['learning_rate']}
        momentum: {config['training_param']['optimizer_param']['momentum']}
criterion:
    type: {loss}
eval:
    freq: {int(config['training_param']['epochs']) + 10}
    metrics: ['{metric}', 'acc']
    best_res_update_round_wise_key: 'test_loss'
data:
    type: ''
model:
    type: '{model_type}'
    input_shape: {input_len}
    {hidden_layer}
    {layer_num}
    out_channels: {num_class}
seed: {np.random.randint(2023)}
"""

    with open('./federatedscope/federatedscope/contrib/configs/server.yaml', 'w') as f:
        f.write(server_config)

    for client_idx in range(1, client_num + 1):
        client_config = f"""use_gpu: False
federate:
    client_num: {client_num}
    sample_client_num: {config['training_param']['client_per_round']}
    mode: 'distributed'
    total_round_num: {config['training_param']['epochs']}
    make_global_eval: False
    online_aggr: False
    data_weighted_aggr: True
distribute:
    use: True
    server_host: '{server_ip}'
    server_port: 50051
    client_host: '{client_ip}'
    client_port: {50051+client_idx}
    role: 'client'
    data_idx: {client_idx}
trainer:
    type: 'general'
train:
    optimizer:
        type: SGD
        lr: {config['training_param']['learning_rate']}
        momentum: {config['training_param']['optimizer_param']['momentum']}
criterion:
    type: {loss}
eval:
    freq: {int(config['training_param']['epochs']) + 10}
    metrics: ['{metric}', 'acc']
    best_res_update_round_wise_key: 'test_loss'
data:
    type: 'file'
    batch_size: {config['training_param']['batch_size']}
model:
    type: '{model_type}'
    input_shape: {input_len}
    {hidden_layer}
    {layer_num}
    out_channels: {num_class}
seed: {np.random.randint(2023)}
"""
        with open(f'./federatedscope/federatedscope/contrib/configs/client_{client_idx}.yaml', 'w') as f:
            f.write(client_config)


def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "federatedscope"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config

def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, server_ip: str, config: dict, client_ip: str):
    # convert config format
    federatedscope_config = copy.deepcopy(config)
    federatedscope_config["training_param"] = federatedscope_config["training"]
    federatedscope_config.pop("training")
    federatedscope_config["training_param"]["epochs"] = federatedscope_config["training_param"]["global_epochs"]
    federatedscope_config["bench_param"] = federatedscope_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(federatedscope_config, cf)
    convert_config(role, server_ip, client_ip)
    with GetTempFileName() as temp_log_filename, \
        GetTempFileName() as temp_output_filename:
        # note that here, you don't have to create temp files to receive output and log
        # you can also expect the target process to generate files and then read them

        # start training procedure
        if config['dataset'].split('_')[-1] == 'vertical':
            if role == "client":
                role_id == "dummy"
            else:
                role_id = "vertical"
        else:
            role_id = role if role == "server" else f"{role}_{str(participant_id)}"
        print(role_id, "begin")
        if role_id == "dummy":
            process = subprocess.Popen(
                [
                    "cd",
                    ".",
                ],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        else:
            process = subprocess.Popen(
                [
                    "python",  
                    # takes 4 args: mode(client/server), participant_id, output, and logging destination
                    "federatedscope/federatedscope/main.py",
                    "--cfg",
                    f"federatedscope/federatedscope/contrib/configs/{role_id}.yaml",
                    # temp_output_filename,
                    # temp_log_filename,
                ],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        # gather result
        stdout, stderr = process.communicate()
        print(role_id, "done")
        returncode = process.returncode
        with open(temp_output_filename, "rb") as f:
            output = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
        with open(temp_log_filename, "rb") as f:
            log = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)
        return json.dumps({
            "server_ip": server_ip,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
        })


@pop.handle("unifed.federatedscope:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    # for certain frameworks, clients need to learn the ip of the server
    # in that case, we get the ip of the current machine and send it to the clients
    server_ip = get_local_ip()
    cl.send_variable("server_ip", server_ip, [p for p in participants if p.role == "client"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "server", server_ip, unifed_config, server_ip)


@pop.handle("unifed.federatedscope:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()
    client_ip = get_local_ip()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "client", server_ip, unifed_config, client_ip)
