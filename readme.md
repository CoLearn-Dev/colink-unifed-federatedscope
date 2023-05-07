# CoLink - UniFed - FederatedScope

This repo integrates FederatedScope, a federated learning framework, into UniFed as a [CoLink](https://colink.app/) protocol. You can follow the steps below to test it.

## 1. Clone the repo

```bash
git clone https://github.com/HenryHu-H/colink-unifed-federatedscope.git
```

## 2. Create an environment

```bash
cd colink-unifed-federatedscope
conda create -n colink-unifed-federatedscope python=3.9
conda activate colink-unifed-federatedscope
```

## 3. Install the package in an editable mode

```bash
pip install -e .
```

## 4. Test the protocol

- The first step is to write a test configuration. You can look into `./test/configs/case_0.json` for an example. Note that for the case you construct, it should mainly serve the purpose of correctness testing (e.g. 1~2 epochs with a small model is usually sufficient). In this way, we can reproduce the correctness testing from a single host.

  - To check the output for running certain cases, change the case string `target_case = "test/configs/case_0.json"` in `test_all_config.py` (note that this only works when you install with `-e` flag), then, in the root directory of the repo, run
```bash
python test/test_all_config.py 
```
- Note that the vertical FL of FederatedScope only supports the standalone mode. Please run the above command first to load the dataset and convert the config file, and then run manually with

```bash
python federatedscope/federatedscope/main.py --cfg federatedscope/federatedscope/contrib/configs/vertical.yaml
```

- Please do not run `pytest` directly as it may trigger other test cases provided by the FederatedScope framework.

