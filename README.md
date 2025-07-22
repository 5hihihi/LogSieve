# LogSieve: Log Parsing with Selective LLM-Based Template Rectification

With the rise of the **Internet of Things (IoT)**, billions of devices generate massive volumes of log data, capturing their opera tions and interactions. These logs are typically messy and unstructured, making analysis difficult. By transforming raw logs into structured tem plates, log parsing provides a fundamental basis for various operational tasks, including fault detection, anomaly diagnosis, and large-scale sys tem monitoring. However, existing methods face limitations: heuristic based approaches struggle with pattern variability, while deep learning models, particularly **large language models (LLMs)**, are computa tionally intensive and unstable for industrial deployment. 

We propose **LogSieve**, a hybrid log parsing framework that balances efficiency and adaptability. Built upon a prefix-tree parser, LogSieve introduces three key components: (1) a semantically weighted similar ity function for enhanced template grouping, (2) a fragmented template rectification module to merge structurally similar templates, and (3) a confidence-aware postprocessing step that selectively invokes LLMs to refine low-confidence results. Experiments on public datasets show that LogSieve improves parsing accuracy by 15.7% while reducing LLM usage by 78.2%. It achieves pro cessing speeds of over 0.4 million lines per minute, demonstrating its scalability and practicality for large-scale log analysis.


## Datasets download

Please first download the full datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861).

Then, you need to put these datasets into `full_dataset/` following the format of `2k_dataset`.


## Repository Organization 

```
├── 2k_dataset/ # the original Loghub-2k datasets
├── full_dataset/ # unzip the Loghub-2.0 into this directory
│   └── post_process.py # we provide the heuristic roles used in our annotation of templates 
├── benchmark/
│   ├── evaluation/
│   ├── logparser/
│   ├── LogSieve_benchmark.py/
│   ├── run_statistic_2k.sh # the script to run all statistic-based log parsers on Loghub-2k datasets
│   └── run_statistic_full.sh # the script to run all statistic-based log parsers on Loghub-2.0 datasets
├── result/
│   ├── ...... # 
│   └── ...... # contains the output evaluation metric files and all parsed results
├── requirements.txt
└── README.MD
```

## Requirements

Owing to the large scale of the benchmark in the experiments, the requirements of the benchmark of all log parsers are:

- At least 16GB memory.
- At least 60GB storage.

**Installation**

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```

### Quick Demo

We give a demo script to run Drain on both Loghub-2k and Loghub-2.0, this will takes about 2-3 hours.

```bash
cd benchmark/
./demo.sh
```
