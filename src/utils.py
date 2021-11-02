import logging
import os
import shutil
import sys
from datetime import datetime


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def setup_logger(args):
    global now
    global logger

    logger = logging.getLogger()
    now = datetime.now()

    time_stamp = now.strftime("%Y%m%d-%H%M%S")


    if args.getboolean('Log', 'log_output'):
        curr_exp_log_dir = args.get('Log', 'logs_main_dir') + 'Log-'+time_stamp+'/'
        os.makedirs(curr_exp_log_dir)
        log = open(curr_exp_log_dir + "logger.log", "a")
        sys.stdout = log

    curr_exp_model_dir = args.get('Model', 'models_main_dir') + 'Model-' + time_stamp + '/'
    os.makedirs(curr_exp_model_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(args.getint('Log', 'log_level'))

    if args.getboolean('Log', 'flush_history'):
        try:
            logger.info("Try to Delete log history")
            for f in os.listdir(args.get('Log', 'logs_main_dir')):
                if f != 'Log-'+time_stamp:
                    shutil.rmtree(args.get('Log', 'logs_main_dir') + f)
        except:
            logger.warning("Path is invalid, could not delete log history !")

    logger.info("All configs dict:")
    print({section: dict(args[section]) for section in args.sections()})

    logger.info("Configs per Sections overview")
    for section in args.sections():
        print("Section: " + section)
        print(dict(args[section]))

