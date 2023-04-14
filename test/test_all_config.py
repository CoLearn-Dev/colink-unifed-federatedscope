import glob
import json
import pytest
import copy
import os
import sys
import numpy as np

import flbenchmark.datasets
import flbenchmark.logging

import colink as CL


def convert_config():
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
    
    FATE_DATASETS_VERTICAL = ['motor_vertical', 'breast_vertical', 'default_credit_vertical',
                            'dvisits_vertical', 'give_credit_vertical', 'student_vertical', 'vehicle_scale_vertical']

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
    server_host: '127.0.0.1'
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
    server_host: '127.0.0.1'
    server_port: 50051
    client_host: '127.0.0.1'
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

def simulate_with_config(config_file_path):
    from unifed.frameworks.federatedscope.protocol import pop, UNIFED_TASK_DIR
    case_name = config_file_path.split("/")[-1].split(".")[0]
    with open(config_file_path, "r") as cf:
        config = json.load(cf)
    # convert config format
    federatedscope_config = copy.deepcopy(config)
    federatedscope_config["training_param"] = federatedscope_config["training"]
    federatedscope_config.pop("training")
    federatedscope_config["bench_param"] = federatedscope_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(federatedscope_config, cf)
    convert_config()

    # use instant server for simulation
    ir = CL.InstantRegistry()
    # TODO: confirm the format of `participants``
    config_participants = config["deployment"]["participants"]
    cls = []
    participants = []
    for _, role in config_participants:  # given user_ids are omitted and we generate new ones here
        cl = CL.InstantServer().get_colink().switch_to_generated_user()
        pop.run_attach(cl)
        participants.append(CL.Participant(user_id=cl.get_user_id(), role=role))
        cls.append(cl)
    task_id = cls[0].run_task("unifed.federatedscope", json.dumps(config), participants, True)
    results = {}
    def G(key):
        r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
        if r is not None:
            if key == "log":
                return [json.loads(l) for l in r.decode().split("\n") if l != ""]
            return r.decode() if key != "return" else json.loads(r)
    for cl in cls:
        cl.wait_task(task_id)
        results[cl.get_user_id()] = {
            "output": G("output"),
            "log": G("log"),
            "return": G("return"),
            "error": G("error"),
        }
    return case_name, results


def test_load_config():
    # load all config files under the test folder
    config_file_paths = glob.glob("test/configs/*.json")
    assert len(config_file_paths) > 0


@pytest.mark.parametrize("config_file_path", glob.glob("test/configs/*.json"))
def test_with_config(config_file_path):
    if "skip" in config_file_path:
        pytest.skip("Skip this test case")
    results = simulate_with_config(config_file_path)
    for r in results[1].values():
        print(r["return"]["stderr"])
    assert all([r["error"] is None and r["return"]["returncode"] == 0 for r in results[1].values()])


if __name__ == "__main__":
    from pprint import pprint
    import time
    nw = time.time()
    target_case = "test/configs/case_0.json"
    # print(json.dumps(simulate_with_config(target_case), indent=2))
    results = simulate_with_config(target_case)
    for r in results[1].values():
        print(r["return"]["stdout"])
        print(r["return"]["stderr"])
    flbenchmark.logging.get_report('./log')
    print("Time elapsed:", time.time() - nw)
