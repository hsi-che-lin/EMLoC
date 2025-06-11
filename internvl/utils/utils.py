import functools
import logging
import sys
import transformers
from transformers.utils.logging import enable_default_handler, set_verbosity


def initLogger(training_args, logger):
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()

    fmt = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    logging.basicConfig(
        format = fmt,
        handlers = [logging.StreamHandler(sys.stdout)],
    )
    formatter = logging.Formatter(fmt)
    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setFormatter(formatter)

    for (name, l) in logging.root.manager.loggerDict.items():
        if isinstance(l, logging.Logger):
            if ("internvl" in name):
                l.addHandler(default_handler)
                l.setLevel(log_level)
                l.propagate = False
            elif ("transformers" in name):
                for handler in l.handlers:
                    handler.setFormatter(formatter)


# from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    if ("." in attr):
        attr2obj, _, attr2set = attr.rpartition('.')
        setattr(rgetattr(obj, attr2obj), attr2set, val)
    else:
        setattr(obj, attr, val)