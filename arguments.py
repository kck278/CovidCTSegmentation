from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name", help="model type", required=True)
    parser.add_argument(
        "-c", "--num-classes", dest="num_classes", help="number of classes to predict", type=int, default=2
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        help="number of epochs for training",
        type=int,
        default=160,
    )

    parser.add_argument(
        "-lr", 
        "--learning-rate", 
        dest="learning_rate", 
        help="Initial learning rate", 
        type=int, 
        default=1e-4
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="batch size for training",
        type=int,
        default=2,
    )
    
    return parser.parse_args()