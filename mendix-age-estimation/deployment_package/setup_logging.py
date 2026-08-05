import logging
import logging.config


def setup_logging(log_level='INFO'):
    """
    Setup and configure the logging

    :param str base_directory: base directory of the app. The 'log' directory is created in here.
    :param str log_path: path to write logs to, relative from the base_directory
    :param str log_level: indicator of how verbose the logging is
    """

    # Configure the logging. We do it here once, after which it will apply everywhere in the project where the logging
    # module is imported
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(levelname)s %(asctime)s - [%(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'level': log_level.upper(),
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'DEBUG',
            },
        }
    }

    # Save our configuration
    logging.config.dictConfig(log_config)

    # Set level names in lower case but starting with a capital and adding fixed padding
    logging.addLevelName(logging.NOTSET, '[Notset]  ')
    logging.addLevelName(logging.DEBUG, '[Debug]   ')
    logging.addLevelName(logging.INFO, '[Info]    ')
    logging.addLevelName(logging.WARNING, '[Warning] ')
    logging.addLevelName(logging.ERROR, '[Error]   ')
    logging.addLevelName(logging.CRITICAL, '[Critical]')