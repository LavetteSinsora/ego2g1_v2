import logging

import tyro

from . import RUNGS

logging.basicConfig(level=logging.INFO, force=True)
tyro.extras.subcommand_cli_from_dict(RUNGS)
