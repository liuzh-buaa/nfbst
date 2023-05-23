"""
    【Merge Evidence】(Optional)
    This is only for accelerating. Chunks run Step 4 in parallel and then merge.
    Note the timestamps to be merged need to be set in the code.

    `python ensemble_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --algorithm p_s --y_index 0`
"""
import shutil
import time

import numpy as np
import pandas as pd

from utils.utils_file import generate_bayes_factors_filename, \
    generate_bayes_factors_excel_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='ensemble model name')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s', choices=['mean', 'mean_abs', 'p_s'])

    opt = parser.parse_args()
    opt.exp_name = 'ensemble_bayes_factors'
    init_config(opt)

    # timestamps = [  # mnist gaussian_e 123 y0
    #     "2022-12-02 20-00-41",  # [0, 1000)
    #     "2022-12-02 20-01-28",  # [1000, 2000)
    #     "2022-12-02 20-01-33",  # [2000, 3000)
    #     "2022-12-02 20-01-41",  # [3000, 4000)
    #     "2022-12-02 20-01-49",  # [4000, 5000)
    #     "2022-12-02 20-01-55",  # [5000, 6000)
    #     "2022-12-02 20-02-02",  # [6000, 7000)
    #     "2022-12-02 20-02-08",  # [7000, 8000)
    #     "2022-12-02 20-02-14",  # [8000, 9000)
    #     "2022-12-02 20-02-19",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y1
    #     "2022-12-03 14-38-46",  # [0, 1000)
    #     "2022-12-03 14-34-23",  # [1000, 2000)
    #     "2022-12-03 14-56-45",  # [2000, 3000)
    #     "2022-12-03 14-47-33",  # [3000, 4000)
    #     "2022-12-03 14-42-58",  # [4000, 5000)
    #     "2022-12-03 14-49-18",  # [5000, 6000)
    #     "2022-12-03 14-36-20",  # [6000, 7000)
    #     "2022-12-03 14-40-47",  # [7000, 8000)
    #     "2022-12-03 14-30-23",  # [8000, 9000)
    #     "2022-12-03 14-44-06",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y2
    #     "2022-12-04 09-19-09",  # [0, 1000)
    #     "2022-12-04 09-18-46",  # [1000, 2000)
    #     "2022-12-04 09-30-09",  # [2000, 3000)
    #     "2022-12-04 09-11-36",  # [3000, 4000)
    #     "2022-12-04 09-31-50",  # [4000, 5000)
    #     "2022-12-04 09-41-11",  # [5000, 6000)
    #     "2022-12-04 09-11-20",  # [6000, 7000)
    #     "2022-12-04 09-23-41",  # [7000, 8000)
    #     "2022-12-04 09-12-16",  # [8000, 9000)
    #     "2022-12-04 09-19-36",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y3
    #     "2022-12-05 06-17-05",  # [0, 1000)
    #     "2022-12-05 06-14-14",  # [1000, 2000)
    #     "2022-12-05 06-30-28",  # [2000, 3000)
    #     "2022-12-05 06-08-58",  # [3000, 4000)
    #     "2022-12-05 06-36-44",  # [4000, 5000)
    #     "2022-12-05 06-52-59",  # [5000, 6000)
    #     "2022-12-05 06-10-58",  # [6000, 7000)
    #     "2022-12-05 06-26-48",  # [7000, 8000)
    #     "2022-12-05 06-15-33",  # [8000, 9000)
    #     "2022-12-05 06-18-46",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y4
    #     "2022-12-06 10-59-48",  # [0, 1000)
    #     "2022-12-06 11-06-09",  # [1000, 2000)
    #     "2022-12-06 11-21-44",  # [2000, 3000)
    #     "2022-12-06 10-48-05",  # [3000, 4000)
    #     "2022-12-06 11-03-09",  # [4000, 5000)
    #     "2022-12-06 11-43-49",  # [5000, 6000)
    #     "2022-12-06 11-03-11",  # [6000, 7000)
    #     "2022-12-06 11-15-22",  # [7000, 8000)
    #     "2022-12-06 10-58-19",  # [8000, 9000)
    #     "2022-12-06 11-16-54",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y5
    #     "2022-12-07 18-12-05",  # [0, 1000)
    #     "2022-12-07 18-36-22",  # [1000, 2000)
    #     "2022-12-07 18-40-36",  # [2000, 3000)
    #     "2022-12-07 17-59-06",  # [3000, 4000)
    #     "2022-12-07 18-23-29",  # [4000, 5000)
    #     "2022-12-07 19-17-18",  # [5000, 6000)
    #     "2022-12-07 18-35-11",  # [6000, 7000)
    #     "2022-12-07 18-33-58",  # [7000, 8000)
    #     "2022-12-07 18-32-05",  # [8000, 9000)
    #     "2022-12-07 18-31-08",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y6
    #     "2022-12-08 23-21-38",  # [0, 1000)
    #     "2022-12-08 23-52-32",  # [1000, 2000)
    #     "2022-12-08 23-46-01",  # [2000, 3000)
    #     "2022-12-08 23-07-49",  # [3000, 4000)
    #     "2022-12-08 23-22-02",  # [4000, 5000)
    #     "2022-12-09 00-06-52",  # [5000, 6000)
    #     "2022-12-08 23-46-45",  # [6000, 7000)
    #     "2022-12-08 23-45-03",  # [7000, 8000)
    #     "2022-12-08 23-32-36",  # [8000, 9000)
    #     "2022-12-08 23-43-23",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y7
    #     "2022-12-06 05-31-03",  # [0, 1000)
    #     "2022-12-06 05-38-20",  # [1000, 2000)
    #     "2022-12-06 05-35-49",  # [2000, 3000)
    #     "2022-12-06 05-23-22",  # [3000, 4000)
    #     "2022-12-06 05-38-32",  # [4000, 5000)
    #     "2022-12-06 05-27-28",  # [5000, 6000)
    #     "2022-12-06 05-25-20",  # [6000, 7000)
    #     "2022-12-06 05-13-49",  # [7000, 8000)
    #     "2022-12-06 05-26-27",  # [8000, 9000)
    #     "2022-12-06 05-36-18",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y8
    #     "2022-12-07 12-52-35",  # [0, 1000)
    #     "2022-12-07 12-54-01",  # [1000, 2000)
    #     "2022-12-07 12-58-40",  # [2000, 3000)
    #     "2022-12-07 13-14-52",  # [3000, 4000)
    #     "2022-12-07 12-39-42",  # [4000, 5000)
    #     "2022-12-07 12-57-21",  # [5000, 6000)
    #     "2022-12-07 12-49-45",  # [6000, 7000)
    #     "2022-12-07 12-40-18",  # [7000, 8000)
    #     "2022-12-07 12-41-50",  # [8000, 9000)
    #     "2022-12-07 12-42-06",  # [9000, 10000)
    # ]
    #
    # timestamps = [  # mnist gaussian_e 123 y9
    #     "2022-12-08 17-15-41",  # [0, 1000)
    #     "2022-12-08 17-33-03",  # [1000, 2000)
    #     "2022-12-08 17-43-06",  # [2000, 3000)
    #     "2022-12-08 17-46-06",  # [3000, 4000)
    #     "2022-12-08 17-06-25",  # [4000, 5000)
    #     "2022-12-08 17-29-14",  # [5000, 6000)
    #     "2022-12-08 17-17-06",  # [6000, 7000)
    #     "2022-12-08 16-51-24",  # [7000, 8000)
    #     "2022-12-08 17-13-06",  # [8000, 9000)
    #     "2022-12-08 17-05-28",  # [9000, 10000)
    # ]

    # timestamps = [  # simulation_v4 LRP p_s
    #     "2023-04-28 22-22-50",  # [0, 3700)
    #     "2023-04-30 12-31-57",  # [3700, 10000)
    # ]

    # timestamps = [  # simulation_v12 LRP p_s
    #     "2023-04-28 22-22-51",  # [0, 3600)
    #     "2023-04-30 12-32-31",  # [3600, 10000)
    # ]

    # timestamps = [  # simulation_v4 DeepSHAP p_s
    #     "2023-05-02 15-43-18",  # [0, 2000)
    #     "2023-05-02 15-43-48",  # [2000, 4000)
    #     "2023-05-02 15-44-08",  # [4000, 6000)
    #     "2023-05-02 15-44-28",  # [6000, 8000)
    #     "2023-05-02 15-44-42",  # [8000, 10000)
    # ]

    # timestamps = [  # simulation_v12 DeepSHAP p_s
    #     "2023-05-02 07-45-08",  # [0, 2000)
    #     "2023-05-02 07-45-26",  # [2000, 4000)
    #     "2023-05-02 07-45-40",  # [4000, 6000)
    #     "2023-05-02 07-45-55",  # [6000, 8000)
    #     "2023-05-02 07-46-11",  # [8000, 10000)
    # ]

    timestamps = [  # simulation_v3 LIME p_s
        '2023-05-17 20-29-38',
        '2023-05-17 20-30-15',
        '2023-05-17 20-30-32',
        '2023-05-17 20-30-47',
        '2023-05-17 20-30-58',
        '2023-05-17 12-31-05',
        '2023-05-17 12-31-16',
        '2023-05-17 12-31-35',
        '2023-05-17 12-31-45',
        '2023-05-17 12-31-55'
    ]

    init_log_dir = opt.log_dir
    data = []
    # starts, ends = [0, 3600], [3599, 9999]
    for i, timestamp in enumerate(timestamps):
        opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
        data.append(np.load(generate_bayes_factors_filename(opt)))
        # data.append(np.load(generate_bayes_factors_cache_filename(opt, starts[i], ends[i])))

    opt.log_dir = init_log_dir

    np_data = np.concatenate(data, axis=0)
    print(f'np_data: {np_data.shape}')
    np.save(generate_bayes_factors_filename(opt), np_data)

    try:
        features = [f'x{i}' for i in range(opt.n_features)]
        writer = pd.ExcelWriter(generate_bayes_factors_excel_filename(opt))
        pd_data = pd.DataFrame(np_data, columns=features)
        pd_data.to_excel(writer, opt.model_name, float_format='%.3f')
        writer.close()
    except TypeError as e:
        print(repr(e))

    if opt.log:
        print(f'==> Copying bayes factors from `timestamp` to `results`...')
        shutil.copyfile(generate_bayes_factors_filename(opt, last=False),
                        generate_bayes_factors_filename(opt, last=True))

        try:
            shutil.copyfile(generate_bayes_factors_excel_filename(opt, last=False),
                            generate_bayes_factors_excel_filename(opt, last=True))
        except FileNotFoundError as e:
            print(repr(e))

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
