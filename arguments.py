from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name", help="model type", default="UNet")
    parser.add_argument(
        "-c", "--num-classes", dest="num_classes", help="number of classes to predict", type=int, default=2
    )

    parser.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        help="resolution of images",
        type=int,
        default=256
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        help="number of epochs for training",
        type=int,
        default=160
    )

    parser.add_argument(
        "-l", 
        "--learning-rate", 
        dest="learning_rate", 
        help="Initial learning rate", 
        type=float, 
        default=1e-4
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="batch size for training",
        type=int,
        default=2
    )

    parser.add_argument(
        "-ext",
        "--extended",
        dest="extended",
        help="use extended dataset",
        type=bool,
        default=False
    )
    
    return parser.parse_args()