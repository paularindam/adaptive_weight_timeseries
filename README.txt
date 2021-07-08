The code provided implements W-Augment, alpha-trimmed Augment and Rand Augment and evaluates these methods on the UCR Archive dataset using the InceptionTime network described in section 4.3.1, https://arxiv.org/abs/2102.08310

Usage Summary
-------------
	1) conda create --name <env> --file requirements.txt
	2) Download UCR 2018 Archive data from http://www.timeseriesclassification.com/ and change Line 385 in the code to change the data path
	3) python3 main.py --run_path <dest_dir> --augment w_augment

This will create a folder <dest_dir> with the output of the run which consists of two folders: UCR_results with the individual results for each model and the resulting ensemble on each dataset, and a summary_results folder which includes the summary metrics of the full run.

------------------------------------------------------------------------------

usage: main.py [-h] [--run_path RUN_PATH] [--n_epochs N_EPOCHS]
                    [--n_iters N_ITERS] [--datasets DATASETS]
                    [--augment {baseline,rand_augment,w_augment,atrim_augment}]
                    [--param_M PARAM_M]

optional arguments:
  -h, --help            show this help message and exit
  --run_path RUN_PATH   (default: ./)
  --n_epochs N_EPOCHS   (default: 1500)
  --n_iters N_ITERS     (default: 5)
  --datasets DATASETS   (default: all)
  --augment {baseline,rand_augment,w_augment,atrim_augment}
                        (default: None)
  --param_M PARAM_M     (default: 10)