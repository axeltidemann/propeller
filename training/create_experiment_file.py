from __future__ import print_function
import os
import stat

experiment_file = 'experiments.sh'
with open(experiment_file, 'w') as _file:
    # MOEs - everything
    # for loss in ['complex', 'simple']:
    #     for mixture_mode in ['fusion', 'single']:
    #         for n_experts in range(1,6):
    #             for hidden_size in [ 256, 512, 1024, 2048 ]:
    #                 outfile = '{}_{}_{}_{}'.format(loss, mixture_mode, n_experts, hidden_size)
    #                 print(' python ~/propeller/training/moe.py ../../train_top90_curated/ ../../test_top90_curated/ --loss {} --mixture_mode {} --n_experts {} --hidden_size {} > {}'.format(loss, mixture_mode, n_experts, hidden_size, outfile), file=_file)


    # MOEs - I found 10 ensembles to be a figure in a paper
    # for hidden_size in [ 256, 512, 1024, 2048 ]:
    #     outfile = '10_experts_hidden_{}'.format(hidden_size)
    #     print(' python ~/propeller/training/moe.py ../../train_top90_curated/ ../../test_top90_curated/ --n_experts 10 --hidden_size {} > {}'.format(hidden_size, outfile), file=_file)

    
    # Ensembles
    # for n_experts in range(1,6):
    #     for hidden_size in [ 256, 512, 1024, 2048 ]:
    #         for text_expert_hidden_size in  [ 256, 512, 1024, 2048 ]:
    #             outfile = 'hidden_{}_text_hidden_{}_n_experts_{}'.format(hidden_size, text_expert_hidden_size, n_experts)
    #             print(' python ~/propeller/training/ensemble.py ../../train_top90_curated/ ../../test_top90_curated/  --n_experts {} --hidden_size {} --text_expert_hidden_size {} > {}'.format(n_experts, hidden_size, text_expert_hidden_size, outfile), file=_file)


    # Classic monolithic network
    # for hidden_size in [ 128, 256, 512, 1024, 2048 ]:
    #     outfile = 'hidden_size_{}'.format(hidden_size)
    #     print(' python ~/propeller/training/text+image_classifier.py ../../train_top90_curated/ ../../test_top90_curated/ --hidden_size {} > {}'.format(hidden_size, outfile), file=_file)


    # Just LSTM
    for hidden_size in [ 128, 256, 512 ]:
        outfile = 'lstm_size_{}'.format(hidden_size)
        print(' python ~/propeller/training/text+image_classifier.py ../../train_top90_curated/ ../../test_top90_curated/ --lstm_size {} --hidden_size 128 --mode text --embedding lstm > {}'.format(hidden_size, outfile), file=_file)

                    
st = os.stat(experiment_file)
os.chmod(experiment_file, st.st_mode | stat.S_IEXEC)
