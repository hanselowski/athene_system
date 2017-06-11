from fnc.models.MultiThreadingFeedForwardMLP import MultiThreadingFeedForwardMLP
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def get_estimator(scorer_type, save_folder=None):
    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

    if scorer_type == 'voting_mlps_hard':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        mlp1 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp2 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp3 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp4 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp5 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)


        clf = VotingClassifier(estimators=[  # ('gb', gb),
            # ('mlp', mlp),
            ('mlp', mlp1),
            ('mlp', mlp2),
            ('mlp', mlp3),
            ('mlp', mlp4),
            ('mlp', mlp5),
        ],  n_jobs=1,
            voting='hard')



    if scorer_type == 'MLP_base':
        clf = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=200, hm_epochs=30, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.01,
                                           hidden_layers=(600, 600, 600), activation_function='relu', save_folder=save_folder, seed=12345)

    return clf